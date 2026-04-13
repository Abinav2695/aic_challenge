#!/usr/bin/env python3
"""
LeRobot Recorder Node — passive data recorder for the C++ data collection node.

Subscribes to the time-synchronized observation and action topics published by
zss_data_cpp::HybridDataCollectorNode and writes a LeRobotDataset to disk.

Why this is correct:
  The C++ node runs ApproximateTime sync on (ControllerState, JointState, WrenchStamped).
  It only acts — and only publishes — when all three sensors have matching timestamps.
  /data_collection/observation  carries the 32-dim vector built from that snapshot.
  /data_collection/action       carries the 13-dim pose+stiffness that was commanded.
  This recorder subscribes to those two topics, so every recorded frame is guaranteed
  to be temporally consistent — no independent re-subscription to sensor topics needed.

Topic map:
  /data_collection/observation   → Float64MultiArray (32-dim, from C++ synced snapshot)
  /data_collection/action        → Float64MultiArray (13-dim, pose+stiffness)
  /data_collection/episode_event → String ("start" / "done")
  /data_collection/scene_metadata → String (JSON, latched)
  /{left,center,right}_camera/image → Image (cameras are independent — no sync needed)

Usage:
    ros2 run zss_data lerobot_recorder_node \
        --repo_id local/aic_insertion_dataset \
        --dataset_root ~/datasets \
        --batch_id batch_001
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String

from zss_data.data_collection import LeRobotRecorder, RecordingConfig

logger = logging.getLogger(__name__)

latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class LeRobotRecorderNode(Node):
    """
    Purely passive recorder. Receives pre-built obs+action vectors from the C++ node
    so it benefits from the C++ ApproximateTime synchronizer without reimplementing it.
    """

    def __init__(self, rec_cfg: RecordingConfig):
        super().__init__("lerobot_recorder")

        self.rec_cfg = rec_cfg
        self.recorder = LeRobotRecorder(rec_cfg)

        # Latest synchronized obs/action from C++ node
        self._last_obs: np.ndarray | None = None  # 32-dim float32
        self._last_action: np.ndarray | None = None  # 13-dim float32
        self._last_images: dict = {}

        self._episode_active = False
        self._shutting_down = False

        # ── Subscriptions ────────────────────────────────────────────────────
        # These two come from the C++ node's synchronized snapshot — the key benefit.
        self.create_subscription(
            Float64MultiArray,
            "/data_collection/observation",
            self._obs_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/data_collection/action",
            self._action_cb,
            10,
        )

        # Episode lifecycle and metadata
        self.create_subscription(
            String,
            "/data_collection/episode_event",
            self._episode_event_cb,
            latched_qos,
        )
        self.create_subscription(
            String,
            "/data_collection/scene_metadata",
            self._scene_meta_cb,
            latched_qos,
        )

        # Cameras are independent streams — no time sync needed for images
        for cam in ("left_camera", "center_camera", "right_camera"):
            self.create_subscription(
                Image,
                f"/{cam}/image",
                lambda msg, c=cam: self._image_cb(c, msg),
                10,
            )

        self.recorder.setup()
        self.get_logger().info("LeRobotRecorderNode ready.")
        self.get_logger().info(
            f"  local_root : {rec_cfg.local_root or '~/.cache/huggingface/lerobot/<repo_id>'}"
        )
        self.get_logger().info(f"  repo_id    : {rec_cfg.repo_id}")
        self.get_logger().info("Waiting for /data_collection/episode_event 'start'...")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _obs_cb(self, msg: Float64MultiArray):
        """Receives 32-dim observation built from C++ time-synced snapshot."""
        self._last_obs = np.array(msg.data, dtype=np.float32)

    def _action_cb(self, msg: Float64MultiArray):
        """Receives 13-dim action (pose + stiffness) published alongside the command."""
        self._last_action = np.array(msg.data, dtype=np.float32)

    def _image_cb(self, cam_name: str, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._last_images[cam_name] = img

    def _scene_meta_cb(self, msg: String):
        import json

        try:
            meta = json.loads(msg.data)
            if meta.get("scene_id"):
                self.rec_cfg.scene_id = meta["scene_id"]
            if meta.get("batch_id"):
                self.rec_cfg.batch_id = meta["batch_id"]
            if meta.get("task"):
                self.rec_cfg.task = meta["task"]
            self.get_logger().info(
                f"Scene metadata: scene={self.rec_cfg.scene_id} batch={self.rec_cfg.batch_id}"
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to parse scene metadata: {e}")

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def _episode_event_cb(self, msg: String):
        if msg.data == "start" and not self._episode_active:
            self.get_logger().info("Episode START — beginning recording.")
            self.recorder.begin_episode()
            self._episode_active = True
            self._record_timer = self.create_timer(0.05, self._record_frame)

        elif msg.data == "done" and self._episode_active:
            self.get_logger().info("Episode DONE — saving dataset.")
            self._record_timer.cancel()
            self._episode_active = False
            self._finish()

    def _record_frame(self):
        """20 Hz — records one frame if obs+action are available."""
        if not self._episode_active:
            return
        if self._last_obs is None or self._last_action is None:
            return  # C++ node hasn't published yet (e.g. still in INIT_SERVICES)

        images = self._last_images if self.rec_cfg.record_images else None
        self.recorder.add_frame(
            obs_state=self._last_obs.copy(),
            action=self._last_action.copy(),
            images=images,
            task=self.rec_cfg.task,
        )

    def _finish(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.recorder.end_episode()
        self.recorder.save_metadata_sidecar(
            {
                "scene_id": self.rec_cfg.scene_id,
                "batch_id": self.rec_cfg.batch_id,
                "task": self.rec_cfg.task,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": "lerobot_recorder_node (C++ synced pipeline)",
            }
        )
        self.recorder.finalize()
        self.get_logger().info("Dataset saved. Shutting down recorder.")
        rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Passive LeRobot recorder — pairs with zss_data_cpp node.",
    )
    parser.add_argument("--repo_id", default="local/aic_insertion_dataset")
    parser.add_argument("--task", default="Insert cable into port")
    parser.add_argument("--scene_id", default="")
    parser.add_argument("--batch_id", default="")
    parser.add_argument(
        "--dataset_root", default=None, help="Root dir; data saved at <dataset_root>/<batch_id>/"
    )
    parser.add_argument(
        "--local_root", default=None, help="Exact dataset path (overrides --dataset_root)"
    )
    parser.add_argument("--no_images", action="store_true")
    parser.add_argument("--image_scale", type=float, default=0.25)

    args, _ = parser.parse_known_args(sys.argv[1:])

    local_root = args.local_root
    if local_root is None and args.dataset_root and args.batch_id:
        local_root = str(Path(args.dataset_root) / args.batch_id)
        print(f" Dataset path: {local_root}")

    rec_cfg = RecordingConfig(
        repo_id=args.repo_id,
        task=args.task,
        scene_id=args.scene_id,
        batch_id=args.batch_id,
        local_root=local_root,
        record_images=not args.no_images,
        image_scale=args.image_scale,
    )

    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=sys.argv)
    node = LeRobotRecorderNode(rec_cfg)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
