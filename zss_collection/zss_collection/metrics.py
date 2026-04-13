"""
EpisodeMetrics — per-episode runtime quality tracking for zss_collection.

Completely decoupled from ROS, LeRobot, and the controller.  The node
creates one instance at startup, calls reset() before each episode, then
calls update() each tick.  At episode end it calls build_outcome() to get
an EpisodeOutcome ready to pass to EpisodeRecorder.end_episode().

Metrics tracked
───────────────
path_length_m           Cumulative Euclidean distance of the TCP (meters).
                        Maps to AIC Tier 2 trajectory efficiency score.

peak_force_n            Peak compensated force magnitude (Newtons).
                        Proxy for AIC Tier 2 force penalty (>20 N for >1 s).

excess_force_duration_s Cumulative seconds where force_n > FORCE_PENALTY_THRESHOLD_N.
                        AIC penalises -12 pts when this exceeds 1.0 s.

off_limit_contacts      Count of /aic/gazebo/contacts/off_limit messages received.
                        AIC penalises -24 pts for ANY contact.  Requires
                        ros_gz_interfaces (only in eval Docker); stays 0 otherwise.

inserted                Whether /scoring/insertion_event was received.

elapsed_s               Wall-clock seconds since reset().
ticks                   Total update() calls since reset().
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geometry_msgs.msg import Pose, WrenchStamped

    from zss_collection.recorder import EpisodeOutcome


# AIC scoring threshold (ScoringTier2.cc kForceThreshold = 20.0 N)
FORCE_PENALTY_THRESHOLD_N: float = 20.0


@dataclass
class EpisodeMetrics:
    """
    Accumulates per-episode quality metrics during a single collection episode.

    Parameters
    ----------
    tick_rate_s : float
        Duration of one control tick in seconds (default 0.05 = 20 Hz).
        Used to convert tick counts to wall-clock durations.

    Usage
    -----
    ::

        metrics = EpisodeMetrics(tick_rate_s=0.05)

        # Before starting each episode:
        metrics.reset()

        # Every control tick:
        metrics.update(tcp_pose, wrench)

        # When /scoring/insertion_event arrives:
        metrics.mark_inserted()

        # When /aic/gazebo/contacts/off_limit arrives (optional):
        metrics.mark_contact()

        # At episode end:
        outcome = metrics.build_outcome(
            label="success",
            final_phase="hold",
            phase_ticks={"approach": 80, "correct": 60, ...},
            final_plug_port_dist_m=0.002,   # from TF lookup; -1.0 if unavailable
        )
    """

    tick_rate_s: float = 0.05

    # ── Internal state (reset per episode) ────────────────────────────────────
    _ticks: int = field(default=0, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _path_length_m: float = field(default=0.0, init=False, repr=False)
    _peak_force_n: float = field(default=0.0, init=False, repr=False)
    _excess_force_duration_s: float = field(default=0.0, init=False, repr=False)
    _off_limit_contacts: int = field(default=0, init=False, repr=False)
    _inserted: bool = field(default=False, init=False, repr=False)

    # Previous TCP position for path-length delta (None until first update)
    _prev_tcp: tuple | None = field(default=None, init=False, repr=False)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all accumulators.  Call once before beginning each episode."""
        self._ticks = 0
        self._start_time = time.monotonic()
        self._path_length_m = 0.0
        self._peak_force_n = 0.0
        self._excess_force_duration_s = 0.0
        self._off_limit_contacts = 0
        self._inserted = False
        self._prev_tcp = None

    def update(
        self,
        tcp_pose: Pose,
        wrench: WrenchStamped | None,
        tare_offset: WrenchStamped | None = None,
    ) -> None:
        """
        Record one control tick.

        Parameters
        ----------
        tcp_pose : geometry_msgs/Pose
            Current TCP pose in base_link (from ControllerState).
        wrench : geometry_msgs/WrenchStamped or None
            Raw FT sensor reading from /fts_broadcaster/wrench.
        tare_offset : geometry_msgs/WrenchStamped or None
            Tare offset from ControllerState.fts_tare_offset.
            When provided the force magnitude is computed the same way as the
            AIC eval engine (ScoringTier2.cc WrenchCallback):
                force = raw_wrench.force - tare_offset.force
            When None, falls back to raw wrench (may be inflated by gripper weight).
        """
        self._ticks += 1

        # ── Path length ───────────────────────────────────────────────────────
        p = tcp_pose.position
        curr_tcp = (p.x, p.y, p.z)
        if self._prev_tcp is not None:
            dx = curr_tcp[0] - self._prev_tcp[0]
            dy = curr_tcp[1] - self._prev_tcp[1]
            dz = curr_tcp[2] - self._prev_tcp[2]
            self._path_length_m += math.sqrt(dx * dx + dy * dy + dz * dz)
        self._prev_tcp = curr_tcp

        # ── Force metrics (tare-subtracted, matching AIC eval engine) ─────────
        if wrench is not None:
            fx = wrench.wrench.force.x
            fy = wrench.wrench.force.y
            fz = wrench.wrench.force.z
            if tare_offset is not None:
                fx -= tare_offset.wrench.force.x
                fy -= tare_offset.wrench.force.y
                fz -= tare_offset.wrench.force.z
            force_n = math.sqrt(fx * fx + fy * fy + fz * fz)

            if force_n > self._peak_force_n:
                self._peak_force_n = force_n

            if force_n > FORCE_PENALTY_THRESHOLD_N:
                self._excess_force_duration_s += self.tick_rate_s

    def mark_inserted(self) -> None:
        """Call when /scoring/insertion_event is received."""
        self._inserted = True

    def mark_contact(self) -> None:
        """Call when /aic/gazebo/contacts/off_limit fires (eval Docker only)."""
        self._off_limit_contacts += 1

    # ── Read-only properties ───────────────────────────────────────────────────

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def path_length_m(self) -> float:
        return self._path_length_m

    @property
    def peak_force_n(self) -> float:
        return self._peak_force_n

    @property
    def excess_force_duration_s(self) -> float:
        return self._excess_force_duration_s

    @property
    def off_limit_contacts(self) -> int:
        return self._off_limit_contacts

    @property
    def inserted(self) -> bool:
        return self._inserted

    @property
    def will_incur_force_penalty(self) -> bool:
        """True if the AIC eval engine would apply the -12 pt force penalty."""
        return self._excess_force_duration_s > 1.0

    @property
    def will_incur_contact_penalty(self) -> bool:
        """True if the AIC eval engine would apply the -24 pt contact penalty."""
        return self._off_limit_contacts > 0

    # ── Build outcome ──────────────────────────────────────────────────────────

    def build_outcome(
        self,
        label: str,
        final_phase: str,
        phase_ticks: dict,
        final_plug_port_dist_m: float = -1.0,
    ) -> EpisodeOutcome:
        """
        Construct an EpisodeOutcome from accumulated metrics.

        Parameters
        ----------
        label : str
            Outcome label: "success", "timed_out", "failed", "skipped",
            "quit", "shutdown".
        final_phase : str
            controller.phase.value at episode end.
        phase_ticks : dict
            Mapping of phase name → tick count from the node.
        final_plug_port_dist_m : float
            Euclidean distance (m) between plug frame and port frame at
            episode end.  Pass -1.0 if the TF lookup failed.

        Returns
        -------
        EpisodeOutcome
            Ready to pass to EpisodeRecorder.end_episode(outcome=...).
        """
        # Import here to avoid circular imports (recorder imports metrics
        # only for the type, not at module level).
        from zss_collection.recorder import EpisodeOutcome

        return EpisodeOutcome(
            inserted_successfully=self._inserted and label == "success",
            outcome=label,
            duration_s=round(self.elapsed_s, 2),
            total_ticks=self._ticks,
            final_phase=final_phase,
            phase_ticks=dict(phase_ticks),
            peak_force_n=round(self._peak_force_n, 3),
            path_length_m=round(self._path_length_m, 4),
            excess_force_duration_s=round(self._excess_force_duration_s, 3),
            final_plug_port_dist_m=round(final_plug_port_dist_m, 4),
            off_limit_contacts=self._off_limit_contacts,
        )


# ══════════════════════════════════════════════════════════════════════════════
# AIC score estimation
# ══════════════════════════════════════════════════════════════════════════════

# AIC Tier 2 constants (from ScoringTier2.cc)
_TIER2_DURATION_MIN_S: float = 5.0  # ≤5 s  → max score
_TIER2_DURATION_MAX_S: float = 60.0  # ≥60 s → 0
_TIER2_DURATION_MAX_PTS: float = 12.0

_TIER2_FORCE_PENALTY_PTS: float = -12.0
_TIER2_FORCE_EXCESS_THRESHOLD_S: float = 1.0  # cumulative seconds over 20 N

_TIER2_CONTACT_PENALTY_PTS: float = -24.0

# Smoothness and efficiency can't be computed without jerk / initial distance.
# They are left as None in the estimate and excluded from the total.
_TIER2_SMOOTHNESS_MAX_PTS: float = 6.0
_TIER2_EFFICIENCY_MAX_PTS: float = 6.0

# Tier 3
_TIER3_INSERTION_PTS: float = 75.0
_TIER3_WRONG_PORT_PTS: float = -12.0


def estimate_aic_score(
    inserted_successfully: bool,
    duration_s: float,
    excess_force_duration_s: float,
    off_limit_contacts: int,
) -> dict:
    """
    Estimate the AIC score from metrics available at collection time.

    Computable components
    ─────────────────────
    tier3_insertion      75 pts if inserted_successfully, else 0
    tier2_duration       0–12 pts (linear, gated on tier3 > 0)
    tier2_force_penalty  –12 pts if excess_force_duration_s > 1.0 s
    tier2_contact_penalty –24 pts if off_limit_contacts > 0

    Not computable (require eval engine)
    ─────────────────────────────────────
    tier2_smoothness     0–6 pts  (needs Savitzky-Golay jerk over rosbag)
    tier2_efficiency     0–6 pts  (needs initial plug-port distance)

    Returns
    -------
    dict with individual components and ``estimated_total``.
    ``estimated_total`` excludes smoothness and efficiency (optimistic upper
    bound; real score will be ≤ this).
    """
    tier3 = _TIER3_INSERTION_PTS if inserted_successfully else 0.0

    # Performance metrics gated on tier3 > 0
    if tier3 > 0:
        t = max(_TIER2_DURATION_MIN_S, min(_TIER2_DURATION_MAX_S, duration_s))
        duration_score = (
            (_TIER2_DURATION_MAX_S - t)
            / (_TIER2_DURATION_MAX_S - _TIER2_DURATION_MIN_S)
            * _TIER2_DURATION_MAX_PTS
        )
    else:
        duration_score = 0.0

    force_penalty = (
        _TIER2_FORCE_PENALTY_PTS
        if excess_force_duration_s > _TIER2_FORCE_EXCESS_THRESHOLD_S
        else 0.0
    )
    contact_penalty = _TIER2_CONTACT_PENALTY_PTS if off_limit_contacts > 0 else 0.0

    estimated_total = round(tier3 + duration_score + force_penalty + contact_penalty, 2)

    return {
        "tier3_insertion": round(tier3, 2),
        "tier2_duration": round(duration_score, 2),
        "tier2_force_penalty": force_penalty,
        "tier2_contact_penalty": contact_penalty,
        # Not computable from runtime metrics — excluded from estimated_total
        "tier2_smoothness": None,
        "tier2_efficiency": None,
        # Pessimistic floor: if smoothness/efficiency both 0 → this is real score
        # Optimistic ceiling: add up to +12 pts if controller is smooth/efficient
        "estimated_total": estimated_total,
        "estimated_total_max": round(
            estimated_total + _TIER2_SMOOTHNESS_MAX_PTS + _TIER2_EFFICIENCY_MAX_PTS, 2
        ),
    }
