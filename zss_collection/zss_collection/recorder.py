"""
LeRobot dataset recorder for the zss_collection pipeline.

Wraps the LeRobotDataset `create → add_frame → save_episode → finalize`
lifecycle and keeps it completely decoupled from ROS and from the controller.

Session model
─────────────
One session = one call to setup() + N episodes + one call to finalize().
The node calls setup() at startup, then loops:
    begin_episode() → add_frame() × N → end_episode()
across as many trials as the batch contains, then calls finalize() on shutdown.

Episode overwrite
─────────────────
When the orchestrator sends `overwrite_episode_index: N` in scene_metadata,
EpisodeRecorder appends the new recording normally (it gets index K), then
replaces episode N with the new data by rewriting the parquet files.
After the swap, `total_episodes` goes back to what it was before recording
(K replaces N, so net effect = replaced, not appended).

Dataset on disk
───────────────
    <dataset_root>/<batch_id>/
        meta/
            info.json                 ← LeRobot metadata (totals, feature shapes)
            episodes/chunk-000/       ← per-episode stats + frame range metadata
            tasks.json                ← task label registry
            custom_metadata.jsonl     ← our sidecar (scene_id, trial_id, …)
        data/chunk-000/               ← observation + action tensors (parquet)
        videos/                       ← optional, when record_images=True

Action representation
─────────────────────
7-dim delta pose in TCP frame [Δx, Δy, Δz, Δqx, Δqy, Δqz, Δqw].
Stiffness is a fixed controller parameter — NOT part of the action space.
This keeps the model output small and board-position-invariant.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# Prevent LeRobot from trying to contact HuggingFace Hub during recording.
# Without this, a missing or partial local dataset triggers a Hub API call
# which fails (401) in offline/airgapped environments.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from zss_collection.action import ActionBuilder
from zss_collection.config import RecordingConfig, SessionConfig, TrialConfig
from zss_collection.controllers.base import Phase
from zss_collection.metrics import estimate_aic_score
from zss_collection.obs import ObservationBuilder

logger = logging.getLogger(__name__)


# ── Episode outcome ────────────────────────────────────────────────────────────


@dataclass
class EpisodeOutcome:
    """
    Per-episode scoring data produced by the node and consumed by the recorder.

    Written to <dataset_root>/meta/scores.jsonl — one JSON line per saved episode.
    The score file is designed to be loaded independently (no LeRobot dependency)
    for filtering training data by quality.

    Outcome labels
    ──────────────
    "success"     insertion_event fired, robot held insertion for hold_ticks
    "timeout"     max_episode_ticks reached without insertion
    "force_limit" peak force exceeded insert_force_limit_n before insertion
    "skipped"     operator pressed [s] — no frames saved (score entry still written)
    "partial"     node shutdown mid-episode
    """

    # Was the /scoring/insertion_event received before timeout?
    inserted_successfully: bool = False

    # Outcome label (see docstring above)
    outcome: str = "partial"

    # Episode wall-clock duration in seconds
    duration_s: float = 0.0

    # Total control ticks executed
    total_ticks: int = 0

    # Phase where the episode ended (e.g. "DESCEND" on timeout)
    final_phase: str = "UNKNOWN"

    # Ticks spent in each phase — useful to detect episodes stuck in CORRECT
    phase_ticks: dict = field(default_factory=dict)

    # Peak compensated force magnitude (Newton)
    peak_force_n: float = 0.0

    # ── AIC Tier 2 metrics (from EpisodeMetrics) ──────────────────────────────

    # Cumulative TCP path length (maps to Tier 2 trajectory efficiency)
    path_length_m: float = 0.0

    # Seconds where force > 20 N (AIC penalty threshold = 1.0 s cumulative)
    excess_force_duration_s: float = 0.0

    # Plug-to-port distance at episode end (-1.0 if TF lookup failed)
    final_plug_port_dist_m: float = -1.0

    # Off-limit contact events (AIC -24 pt penalty for any contact)
    off_limit_contacts: int = 0


class EpisodeRecorder:
    """
    Multi-episode LeRobot dataset recorder for one collection session.

    Lifecycle
    ─────────
        recorder = EpisodeRecorder(session_cfg, recording_cfg)
        recorder.setup()                                       # once at startup

        for each trial:
            recorder.begin_episode(trial_config)
            while not done:
                recorder.add_frame(obs, action, phase, images, task)
            recorder.end_episode()

        recorder.finalize()                                    # once at shutdown
    """

    def __init__(self, session_cfg: SessionConfig, recording_cfg: RecordingConfig) -> None:
        self._session_cfg = session_cfg
        self._recording_cfg = recording_cfg
        self._dataset = None
        self._episode_active = False
        self._frame_count = 0
        self._current_trial: TrialConfig | None = None
        # trial_id → episode_index, built from sidecar at setup() time
        self._trial_id_to_episode: dict[str, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        Create (or resume) a LeRobotDataset on disk.

        Safe to call when a dataset already exists at local_root — it will
        resume so that episodes from this session are appended.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        cfg = self._recording_cfg
        root_path = Path(self._session_cfg.local_root)

        features = {
            "observation.state": ObservationBuilder.lerobot_feature_spec(),
            "action": ActionBuilder.lerobot_feature_spec(),
        }

        if cfg.record_images:
            scaled_h = int(cfg.image_height * cfg.image_scale)
            scaled_w = int(cfg.image_width * cfg.image_scale)
            for cam in ("left_camera", "center_camera", "right_camera"):
                features[f"observation.images.{cam}"] = {
                    "dtype": "video",
                    "shape": (scaled_h, scaled_w, 3),
                    "names": ["height", "width", "channels"],
                }

        # A complete LeRobot dataset requires both info.json AND tasks.parquet.
        # If only info.json exists the dataset was partially created (e.g. the
        # process crashed before the first episode was saved). Trying to resume
        # such a dataset causes LeRobotDatasetMetadata.load_metadata() to throw
        # FileNotFoundError, which falls into a catch block that then tries to
        # contact HuggingFace — which fails for local/ repo_ids.
        # Treat a partial dataset as non-existent and recreate cleanly.
        info_path = root_path / "meta" / "info.json"
        tasks_path = root_path / "meta" / "tasks.parquet"
        dataset_complete = info_path.exists() and tasks_path.exists()

        if dataset_complete:
            logger.info("EpisodeRecorder: resuming existing dataset at %s", root_path)
            self._dataset = LeRobotDataset(
                repo_id=cfg.repo_id,
                root=root_path,
            )
        else:
            if info_path.exists() and not tasks_path.exists():
                logger.warning(
                    "EpisodeRecorder: found partial dataset at %s (info.json but no "
                    "tasks.parquet) — removing and recreating cleanly.",
                    root_path,
                )
                import shutil

                shutil.rmtree(root_path)
            logger.info("EpisodeRecorder: creating new dataset at %s", root_path)
            self._dataset = LeRobotDataset.create(
                repo_id=cfg.repo_id,
                fps=cfg.fps,
                robot_type=cfg.robot_type,
                features=features,
                root=root_path,
            )

        self._trial_id_to_episode = self._load_sidecar_index(root_path)
        logger.info(
            "EpisodeRecorder: ready — %d existing episodes, %d trial IDs indexed",
            self._dataset.meta.total_episodes,
            len(self._trial_id_to_episode),
        )

    def _load_sidecar_index(self, root_path: Path) -> dict[str, int]:
        """
        Read custom_metadata.jsonl and build a trial_id → episode_index map.

        Only the LAST entry for each trial_id is kept (in case a trial was
        previously overwritten — the most recent recording wins).
        """
        sidecar = root_path / "meta" / "custom_metadata.jsonl"
        index: dict[str, int] = {}
        if not sidecar.exists():
            return index
        with sidecar.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    tid = entry.get("trial_id", "")
                    ep = entry.get("episode_index")
                    if tid and ep is not None:
                        index[tid] = int(ep)
                except Exception:
                    pass
        return index

    def begin_episode(self, trial_config: TrialConfig) -> None:
        """
        Mark the start of a new episode.

        Parameters
        ----------
        trial_config : TrialConfig
            Per-trial metadata (scene_id, target_module, cable_name, …).
            Saved in the sidecar when end_episode() is called.
            If trial_config.overwrite_index is set, end_episode() will replace
            that episode index rather than appending a new one.
        """
        if self._dataset is None:
            raise RuntimeError("Call setup() before begin_episode()")

        # Auto-detect overwrite: if this trial_id was recorded before, overwrite
        # that episode rather than appending a new one. Metadata param takes
        # precedence if explicitly set; otherwise fall back to sidecar lookup.
        trial = trial_config
        if trial.overwrite_index is None and trial.trial_id in self._trial_id_to_episode:
            existing_idx = self._trial_id_to_episode[trial.trial_id]
            if existing_idx < self._dataset.meta.total_episodes:
                trial = TrialConfig(
                    **{
                        k: getattr(trial_config, k)
                        for k in trial_config.__dataclass_fields__
                        if k != "overwrite_index"
                    },
                    overwrite_index=existing_idx,
                )
                logger.info(
                    "EpisodeRecorder: trial_id '%s' already recorded as episode %d — will overwrite",
                    trial.trial_id,
                    existing_idx,
                )

        self._episode_active = True
        self._frame_count = 0
        self._current_trial = trial
        logger.info(
            "EpisodeRecorder: episode started — scene=%s port=%s/%s%s",
            trial.scene_id,
            trial.target_module,
            trial.port_name,
            f" (will overwrite episode {trial.overwrite_index})"
            if trial.overwrite_index is not None
            else " (new episode)",
        )

    def add_frame(
        self,
        obs_state: np.ndarray,
        action_vec: np.ndarray,
        phase: Phase,
        images: dict | None = None,
        task: str | None = None,
    ) -> None:
        """
        Record one control timestep.

        Parameters
        ----------
        obs_state : np.ndarray, shape (32,), dtype float32
            Observation vector from ObservationBuilder.build().
        action_vec : np.ndarray, shape (7,), dtype float32
            Delta pose action from ActionBuilder.from_poses().
        phase : Phase
            Current controller phase — stored as the LeRobot task string so
            episodes can be filtered by phase in training.
        images : dict, optional
            Mapping of camera_name → np.ndarray (H, W, 3) uint8.
            Only used when RecordingConfig.record_images is True.
        task : str, optional
            Natural-language task override (e.g. from TrialConfig.task).
            If None, the phase label is used as the task string.
        """
        if not self._episode_active or self._dataset is None:
            return

        frame: dict = {
            "observation.state": torch.from_numpy(obs_state).float(),
            "action": torch.from_numpy(action_vec).float(),
            "task": task if task is not None else phase.value,
        }

        if images and self._recording_cfg.record_images:
            scale = self._recording_cfg.image_scale
            for cam_name, img_np in images.items():
                if scale != 1.0:
                    from PIL import Image as PILImage

                    h, w = img_np.shape[:2]
                    img_np = np.array(
                        PILImage.fromarray(img_np).resize(
                            (int(w * scale), int(h * scale)),
                            PILImage.LANCZOS,
                        )
                    )
                frame[f"observation.images.{cam_name}"] = torch.from_numpy(img_np)

        self._dataset.add_frame(frame)
        self._frame_count += 1

    def end_episode(self, outcome: EpisodeOutcome | None = None) -> None:
        """
        Flush the current episode to disk, write a per-episode sidecar, and
        append a score entry to scores.jsonl.

        Parameters
        ----------
        outcome : EpisodeOutcome, optional
            Runtime stats (insertion success, timing, peak force, phase ticks).
            When None a minimal "partial" outcome is used (e.g. on shutdown).
            If `trial_config.overwrite_index` was set in begin_episode(), the newly
            recorded episode replaces that index in the parquet files rather than
            being appended as a new episode.

        Safe to call even if no frames were recorded (e.g. aborted episode).
        """
        if not self._episode_active or self._dataset is None:
            return

        if self._frame_count == 0:
            logger.warning("EpisodeRecorder: end_episode() called with 0 frames — skipping save")
            self._episode_active = False
            return

        self._dataset.save_episode()
        self._episode_active = False

        new_idx = self._dataset.meta.total_episodes - 1
        target_idx = (
            self._current_trial.overwrite_index if self._current_trial is not None else None
        )

        # episode_index to record in the sidecar: target when overwriting, new_idx otherwise.
        sidecar_idx = target_idx if (target_idx is not None and target_idx != new_idx) else new_idx

        if target_idx is not None and target_idx != new_idx:
            if target_idx >= self._dataset.meta.total_episodes:
                logger.warning(
                    "EpisodeRecorder: overwrite_index=%d is out of range "
                    "(total_episodes=%d) — keeping as new episode %d",
                    target_idx,
                    self._dataset.meta.total_episodes,
                    new_idx,
                )
                target_idx = None
            else:
                logger.info(
                    "EpisodeRecorder: overwriting episode %d with new data "
                    "(recorded as episode %d, %d frames)",
                    target_idx,
                    new_idx,
                    self._frame_count,
                )
                self._do_overwrite(target_idx)
                logger.info(
                    "EpisodeRecorder: overwrite complete — total episodes: %d",
                    self._dataset.meta.total_episodes,
                )
        else:
            logger.info(
                "EpisodeRecorder: episode %d saved — %d frames (total episodes: %d)",
                new_idx,
                self._frame_count,
                self._dataset.meta.total_episodes,
            )

        self._append_metadata_sidecar(episode_index=sidecar_idx, overwrote_index=target_idx)
        self._append_score(episode_index=sidecar_idx, outcome=outcome)

        # Keep in-memory index current so re-runs within the same session also overwrite.
        if self._current_trial and self._current_trial.trial_id:
            self._trial_id_to_episode[self._current_trial.trial_id] = sidecar_idx

    def discard_episode(self, outcome: EpisodeOutcome | None = None) -> None:
        """
        Discard the current episode without saving to disk.

        Called when the operator presses [s] to skip a bad scene.
        Any frames buffered in the LeRobot writer for this episode are dropped
        by calling `clear_episode_buffer()` if available, otherwise by
        reinitialising the dataset's episode buffer state.

        A score entry with outcome="skipped" is still written so the score file
        has a complete record of every trial that was attempted.

        Safe to call if no episode is active.
        """
        if not self._episode_active or self._dataset is None:
            return

        logger.warning(
            "EpisodeRecorder: discarding episode — %d frames dropped",
            self._frame_count,
        )

        # LeRobot ≥0.4 stores frames in an in-memory buffer until save_episode().
        # Calling clear_episode_buffer() or resetting internal state drops them.
        if hasattr(self._dataset, "clear_episode_buffer"):
            self._dataset.clear_episode_buffer()
        elif hasattr(self._dataset, "episode_buffer"):
            # Fallback: overwrite the buffer directly.
            self._dataset.episode_buffer = {}

        # Write a score entry even for discarded episodes (no episode_index since nothing saved).
        skip_outcome = outcome or EpisodeOutcome(
            inserted_successfully=False,
            outcome="skipped",
        )
        self._append_score(episode_index=None, outcome=skip_outcome)

        self._episode_active = False
        self._frame_count = 0
        self._current_trial = None

    def finalize(self) -> None:
        """
        Close parquet writers and compute normalisation statistics.

        Call once at the end of the session, after all episodes are saved.
        Do NOT call finalize() between episodes — it closes the writers.
        """
        if self._dataset is None:
            return
        self._dataset.finalize()
        logger.info("EpisodeRecorder: dataset finalized (stats computed, writers closed)")

    # ── Sidecar ───────────────────────────────────────────────────────────────

    def _append_metadata_sidecar(
        self, episode_index: int | None = None, overwrote_index: int | None = None
    ) -> None:
        """
        Append a JSON line to <root>/meta/custom_metadata.jsonl with
        per-trial metadata that LeRobot's schema does not natively support.

        Using .jsonl (one JSON object per line) so multiple episodes can be
        appended incrementally without loading the whole file.

        Parameters
        ----------
        overwrote_index : int, optional
            If this episode replaced an existing one, set to that episode index.
            Written to the sidecar so downstream tools know which episode was replaced.
        """
        if self._dataset is None:
            return

        trial = self._current_trial
        if episode_index is None:
            episode_index = self._dataset.meta.total_episodes - 1

        entry = {
            "episode_index": episode_index,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_count": self._frame_count,
            "controller": self._session_cfg.control_mode,
            "scene_id": trial.scene_id if trial else "",
            "trial_id": trial.trial_id if trial else "",
            "batch_id": trial.batch_id if trial else "",
            "target_module": trial.target_module if trial else "",
            "port_name": trial.port_name if trial else "",
            "cable_name": trial.cable_name if trial else "",
            "plug_type": trial.plug_type if trial else "",
            "task": trial.task if trial else "",
        }

        if overwrote_index is not None:
            entry["overwrote_episode"] = overwrote_index

        sidecar_path = Path(self._session_cfg.local_root) / "meta" / "custom_metadata.jsonl"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("a") as sidecar_file:
            sidecar_file.write(json.dumps(entry) + "\n")

        logger.debug("EpisodeRecorder: sidecar appended → episode %d", episode_index)

    def _append_score(
        self,
        episode_index: int | None,
        outcome: EpisodeOutcome | None,
    ) -> None:
        """
        Append one JSON line to <root>/meta/scores.jsonl.

        The score file is intentionally minimal and self-contained — it can be
        loaded without LeRobot, pandas, or any framework, purely for deciding
        which episodes to include in training:

            import json
            scores = [json.loads(l) for l in open("meta/scores.jsonl")]
            good = [s["episode_index"] for s in scores if s["inserted_successfully"]]

        Parameters
        ----------
        episode_index : int or None
            LeRobot episode index, None for skipped/discarded episodes.
        outcome : EpisodeOutcome or None
            Runtime stats from the node. Uses a minimal partial entry if None.
        """
        trial = self._current_trial
        o = outcome or EpisodeOutcome(inserted_successfully=False, outcome="partial")

        entry: dict = {
            "episode_index": episode_index,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "outcome": o.outcome,
            "inserted_successfully": o.inserted_successfully,
            "duration_s": round(o.duration_s, 2),
            "total_ticks": o.total_ticks,
            "final_phase": o.final_phase,
            "phase_ticks": o.phase_ticks,
            # Force metrics
            "peak_force_n": round(o.peak_force_n, 3),
            "excess_force_duration_s": round(o.excess_force_duration_s, 3),
            # Trajectory metrics (AIC Tier 2)
            "path_length_m": round(o.path_length_m, 4),
            "final_plug_port_dist_m": round(o.final_plug_port_dist_m, 4),
            # Contact penalty indicator
            "off_limit_contacts": o.off_limit_contacts,
            # AIC score estimate (computable at collection time)
            "aic_score": estimate_aic_score(
                inserted_successfully=o.inserted_successfully,
                duration_s=o.duration_s,
                excess_force_duration_s=o.excess_force_duration_s,
                off_limit_contacts=o.off_limit_contacts,
            ),
            # Trial identity for cross-referencing with custom_metadata.jsonl
            "trial_id": trial.trial_id if trial else "",
            "scene_id": trial.scene_id if trial else "",
            "batch_id": trial.batch_id if trial else "",
            "port_name": trial.port_name if trial else "",
            "plug_type": trial.plug_type if trial else "",
        }

        score_path = Path(self._session_cfg.local_root) / "meta" / "scores.jsonl"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with score_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        score_est = entry["aic_score"]
        logger.info(
            "EpisodeRecorder: score written — ep=%s outcome=%s inserted=%s "
            "duration=%.1fs phase=%s peak_force=%.1fN path=%.3fm plug_dist=%.4fm "
            "estimated_score=%.1f (max=%.1f)",
            episode_index,
            o.outcome,
            o.inserted_successfully,
            o.duration_s,
            o.final_phase,
            o.peak_force_n,
            o.path_length_m,
            o.final_plug_port_dist_m,
            score_est["estimated_total"],
            score_est["estimated_total_max"],
        )

    # ── Episode overwrite ─────────────────────────────────────────────────────

    def _do_overwrite(self, target_idx: int) -> None:
        """
        Replace episode `target_idx` in the dataset with the last episode.

        The caller must have already called `dataset.save_episode()` so the
        new data is at episode index `total_episodes - 1`.  After this call
        the dataset is re-opened so subsequent episodes append correctly.

        Steps
        ─────
        1. Finalize the dataset to flush all writers.
        2. Rewrite data parquets: drop target, rename new_idx → target_idx,
           renumber the global `index` column, compact to a single shard.
        3. Rewrite episode-metadata parquets: same drop/rename, recompute
           dataset_from_index / dataset_to_index, compact to single shard.
        4. Patch info.json: total_episodes and total_frames.
        5. Delete stats.json (stale after structural change).
        6. Re-open the dataset so the in-memory state is consistent.
        """
        assert self._dataset is not None
        cfg = self._recording_cfg
        root = Path(self._session_cfg.local_root)

        new_idx = self._dataset.meta.total_episodes - 1

        # Flush all parquet writers before touching files on disk.
        self._dataset.finalize()

        _replace_episode_in_parquets(root, target_idx=target_idx, new_idx=new_idx)

        # Re-open so the in-memory LeRobotDataset reflects the rewritten files.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._dataset = LeRobotDataset(repo_id=cfg.repo_id, root=root)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def total_episodes(self) -> int:
        """Total episodes saved to disk (including prior sessions)."""
        if self._dataset is None:
            return 0
        return self._dataset.meta.total_episodes

    @property
    def is_episode_active(self) -> bool:
        return self._episode_active


# ══════════════════════════════════════════════════════════════════════════════
# Episode overwrite — parquet manipulation
# ══════════════════════════════════════════════════════════════════════════════


def _replace_episode_in_parquets(root: Path, target_idx: int, new_idx: int) -> None:
    """
    Replace episode `target_idx` with the data currently stored as episode `new_idx`.

    This is a pure file-system operation with no ROS or LeRobot dataset object
    dependencies.  The caller must ensure all writers are flushed before calling.

    Layout (LeRobot ≥ 0.4):
        data/chunk-{c:03d}/file-{f:03d}.parquet   — frame rows
        meta/episodes/chunk-{c:03d}/file-{f:03d}.parquet — episode stats rows

    The function compacts every shard into a single file each
    (chunk-000/file-000.parquet) to keep the post-overwrite layout simple.

    Parameters
    ----------
    root : Path
        Dataset root directory.
    target_idx : int
        Episode index to replace (must be < new_idx).
    new_idx : int
        Episode index of the freshly recorded episode (must equal total_episodes - 1).
    """
    import pandas as pd

    # ── 1. Rewrite data frames ─────────────────────────────────────────────

    data_dir = root / "data"
    data_parquets = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not data_parquets:
        raise FileNotFoundError(f"No data parquets found under {data_dir}")

    frames_df = pd.concat(
        [pd.read_parquet(p) for p in data_parquets],
        ignore_index=True,
    )

    # Count frames for each episode before mutation (needed for info.json patch).
    target_frame_count = int((frames_df["episode_index"] == target_idx).sum())
    new_frame_count = int((frames_df["episode_index"] == new_idx).sum())

    # Drop the old episode, then rename the new episode to its slot.
    frames_df = frames_df[frames_df["episode_index"] != target_idx].copy()
    frames_df.loc[frames_df["episode_index"] == new_idx, "episode_index"] = target_idx

    # Re-sort so frames are ordered: episodes 0..N-2 keep their order, with
    # the new data for target_idx placed where target_idx would naturally sit.
    frames_df = frames_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # Renumber the global `index` column (must be 0..total_frames-1, contiguous).
    frames_df["index"] = range(len(frames_df))

    # Delete old shards and write a single compacted shard.
    for p in data_parquets:
        p.unlink()
    out_data_dir = data_dir / "chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    frames_df.to_parquet(out_data_dir / "file-000.parquet", index=False)

    logger.info(
        "_replace_episode_in_parquets: data rewritten — "
        "replaced %d frames with %d frames, total=%d",
        target_frame_count,
        new_frame_count,
        len(frames_df),
    )

    # ── 2. Rewrite episode metadata ────────────────────────────────────────

    meta_ep_dir = root / "meta" / "episodes"
    ep_parquets = sorted(meta_ep_dir.glob("chunk-*/file-*.parquet"))
    if not ep_parquets:
        raise FileNotFoundError(f"No episode-metadata parquets found under {meta_ep_dir}")

    episodes_df = pd.concat(
        [pd.read_parquet(p) for p in ep_parquets],
        ignore_index=True,
    )

    # Drop target row; rename new_idx row to target_idx.
    episodes_df = episodes_df[episodes_df["episode_index"] != target_idx].copy()
    episodes_df.loc[episodes_df["episode_index"] == new_idx, "episode_index"] = target_idx

    # Re-sort by episode_index so ordering is consistent.
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)

    # Recompute cumulative dataset_from_index / dataset_to_index from `length`.
    lengths = episodes_df["length"].to_numpy()
    from_indices = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(int)
    to_indices = np.cumsum(lengths).astype(int)
    episodes_df["dataset_from_index"] = from_indices
    episodes_df["dataset_to_index"] = to_indices

    # All episodes now live in a single shard — update shard tracking columns if present.
    for col in (
        "data/chunk_index",
        "data/file_index",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
    ):
        if col in episodes_df.columns:
            episodes_df[col] = 0

    # Delete old shards and write compacted file.
    for p in ep_parquets:
        p.unlink()
    out_meta_dir = meta_ep_dir / "chunk-000"
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(out_meta_dir / "file-000.parquet", index=False)

    logger.info(
        "_replace_episode_in_parquets: episode metadata rewritten — %d episodes",
        len(episodes_df),
    )

    # ── 3. Patch info.json ────────────────────────────────────────────────

    info_path = root / "meta" / "info.json"
    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)

        # total_episodes: we removed target and renamed new → target, net -1
        info["total_episodes"] = max(0, info.get("total_episodes", 0) - 1)

        # total_frames: subtract old episode's frames (new frames already counted)
        info["total_frames"] = max(0, info.get("total_frames", 0) - target_frame_count)

        # Update splits so the train split matches the new total_frames.
        total_frames = info["total_frames"]
        if "splits" in info and "train" in info["splits"]:
            info["splits"]["train"] = f"0:{total_frames}"

        with info_path.open("w") as f:
            json.dump(info, f, indent=2)

        logger.info(
            "_replace_episode_in_parquets: info.json patched — total_episodes=%d total_frames=%d",
            info["total_episodes"],
            info["total_frames"],
        )

    # ── 4. Invalidate stale stats ─────────────────────────────────────────

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats_path.unlink()
        logger.info("_replace_episode_in_parquets: deleted stale stats.json")
