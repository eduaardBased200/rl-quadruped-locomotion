from __future__ import annotations
from isaaclab.utils import math as math_utils
import math
import numpy as np
import trimesh

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from collections.abc import Sequence
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg, SubTerrainBaseCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
)
from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfPyramidSlopedTerrainCfg,
    HfWaveTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg


# =============================================================================
# IMU SENSOR NOISE
# =============================================================================

IMU_GYRO_NOISE_STD = 0.015       # rad/s, per timestep
IMU_GYRO_BIAS_STD = 0.03         # rad/s, per episode

IMU_ACCEL_NOISE_STD = 0.08       # m/s^2, per timestep
IMU_ACCEL_BIAS_STD = 0.15        # m/s^2, per episode

# =============================================================================
# ADVANCED IMU REALISM
# =============================================================================

# ---------------------------------------------------------------------------
# 1. BIAS DRIFT / RANDOM WALK
# ---------------------------------------------------------------------------
# Bias-ul nu rămâne perfect constant pe durata episodului.
# Aceste valori controlează cât de repede poate deriva.
#
# IMPORTANT:
# sunt intenționat destul de largi pentru domain randomization.

IMU_GYRO_BIAS_DRIFT_STD = 0.0025       # rad/s drift step
IMU_ACCEL_BIAS_DRIFT_STD = 0.015       # m/s^2 drift step


# ---------------------------------------------------------------------------
# 2. SCALE FACTOR ERROR
# ---------------------------------------------------------------------------
# Fiecare axă a fiecărui IMU poate avea un gain ușor diferit.
#
# Exemplu:
#   true = 2.0
#   scale = 1.04
#   measured ~= 2.08
#
# Randomizat independent pe X/Y/Z și per environment.

IMU_GYRO_SCALE_MIN = 0.90
IMU_GYRO_SCALE_MAX = 1.10

IMU_ACCEL_SCALE_MIN = 0.90
IMU_ACCEL_SCALE_MAX = 1.10


# ---------------------------------------------------------------------------
# 3. AXIS MISALIGNMENT / MOUNTING ORIENTATION ERROR
# ---------------------------------------------------------------------------
# IMU-ul real nu este montat perfect aliniat cu body frame-ul robotului.
# Valorile sunt în grade și vor fi convertite în radiani.

IMU_MISALIGN_ROLL_MIN_DEG = -3.0
IMU_MISALIGN_ROLL_MAX_DEG = +3.0

IMU_MISALIGN_PITCH_MIN_DEG = -3.0
IMU_MISALIGN_PITCH_MAX_DEG = +3.0

IMU_MISALIGN_YAW_MIN_DEG = -3.0
IMU_MISALIGN_YAW_MAX_DEG = +3.0

IMU_MISALIGN_ROLL_MIN_RAD = math.radians(IMU_MISALIGN_ROLL_MIN_DEG)
IMU_MISALIGN_ROLL_MAX_RAD = math.radians(IMU_MISALIGN_ROLL_MAX_DEG)

IMU_MISALIGN_PITCH_MIN_RAD = math.radians(IMU_MISALIGN_PITCH_MIN_DEG)
IMU_MISALIGN_PITCH_MAX_RAD = math.radians(IMU_MISALIGN_PITCH_MAX_DEG)

IMU_MISALIGN_YAW_MIN_RAD = math.radians(IMU_MISALIGN_YAW_MIN_DEG)
IMU_MISALIGN_YAW_MAX_RAD = math.radians(IMU_MISALIGN_YAW_MAX_DEG)


# ---------------------------------------------------------------------------
# 4. SENSOR POSITION OFFSET
# ---------------------------------------------------------------------------
# Poziția reală a IMU-ului față de poziția nominală.
# Începem cu ±3 cm pe fiecare axă.

IMU_POSITION_OFFSET_MIN_M = -0.03
IMU_POSITION_OFFSET_MAX_M = +0.03


# ---------------------------------------------------------------------------
# 5. SENSOR LATENCY
# ---------------------------------------------------------------------------
# Delay-ul este exprimat în physics steps.
#
# IMU delay buffer is applied at the 50 Hz policy-observation rate:
#   1 step = 20 ms
# Current randomized range: 0..1 policy steps.

IMU_DELAY_MIN_STEPS = 0
IMU_DELAY_MAX_STEPS = 1


# ---------------------------------------------------------------------------
# 6. SENSOR SATURATION
# ---------------------------------------------------------------------------
# Limite randomizate de măsurare.
#
# Acestea nu sunt limitele robotului, ci limitele electronice ale senzorului.

IMU_GYRO_SATURATION_MIN = 6.0
IMU_GYRO_SATURATION_MAX = 12.0

IMU_ACCEL_SATURATION_MIN = 30.0
IMU_ACCEL_SATURATION_MAX = 60.0


class FilteredJointPositionAction(JointPositionAction):
    """Joint position action with Exponential Moving Average (EMA) smoothing."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.alpha = cfg.alpha
        self.max_delta = cfg.max_delta
        self.deadband = cfg.deadband

        self._limited_actions = torch.zeros(
            (env.num_envs, self.action_dim),
            device=env.device,
        )

        self._filtered_actions = torch.zeros(
            (env.num_envs, self.action_dim),
            device=env.device,
        )

        self._applied_actions = torch.zeros(
            (env.num_envs, self.action_dim),
            device=env.device,
        )



    def process_actions(self, actions: torch.Tensor):

        # =========================================================
        # 1. SLEW-RATE LIMIT
        # =========================================================
        delta = actions - self._limited_actions

        delta = torch.clamp(
            delta,
            min=-self.max_delta,
            max=self.max_delta,
        )

        self._limited_actions = self._limited_actions + delta

        # =========================================================
        # 2. EMA LOW-PASS FILTER
        # =========================================================
        self._filtered_actions = (
            self.alpha * self._limited_actions
            + (1.0 - self.alpha) * self._filtered_actions
        )

        # =========================================================
        # 3. TRUE ACTION DEADBAND
        # =========================================================
        # Ignorăm doar schimbările foarte mici față de ultima
        # comandă efectiv aplicată.
        #
        # deadband este în unități NORMALIZATE.
        # =========================================================
        command_delta = (
            self._filtered_actions
            - self._applied_actions
        )

        update_mask = torch.abs(command_delta) >= self.deadband

        self._applied_actions = torch.where(
            update_mask,
            self._filtered_actions,
            self._applied_actions,
        )

        # =========================================================
        # 4. SEND TO ACTUATOR
        # =========================================================
        super().process_actions(self._applied_actions)

       

    def reset(self, env_ids=None):
        super().reset(env_ids)

        if env_ids is None:
            self._limited_actions.zero_()
            self._filtered_actions.zero_()
            self._applied_actions.zero_()
        else:
            self._limited_actions[env_ids] = 0.0
            self._filtered_actions[env_ids] = 0.0
            self._applied_actions[env_ids] = 0.0



@configclass
class FilteredJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for filtered joint position action."""

    class_type: type = FilteredJointPositionAction

    alpha: float = 1.0
    max_delta: float = 2.0
    deadband: float = 0.0


# =============================================================================
# CONSTANTE
# =============================================================================

USD_PATH = r"D:\USDCONVERTOR\RobotDog_USD\OMGCHIARFUNCTIONEAZA_CONTACTREADY.usd"


# Backlash is declared before it is consumed below (module executes top-to-bottom).
STS3215_BACKLASH_MAX_DEG = 0.5  # ST-3215-C018 manufacturer maximum
BACKLASH_MIN_DEG = 0.0
BACKLASH_MAX_DEG = STS3215_BACKLASH_MAX_DEG

BACKLASH_MIN_RAD = math.radians(BACKLASH_MIN_DEG)
BACKLASH_MAX_RAD = math.radians(BACKLASH_MAX_DEG)


# Physics / policy timing.
PHYSICS_HZ = 200.0
PHYSICS_DT = 1.0 / PHYSICS_HZ
DECIMATION = 4
POLICY_HZ = PHYSICS_HZ / DECIMATION
POLICY_DT = PHYSICS_DT * DECIMATION
EPISODE_LENGTH_S = 12.0

# ---------------------------------------------------------------------------
# EVENT-DRIVEN LOCOMOTION (NO GLOBAL GAIT CLOCK)
# ---------------------------------------------------------------------------
# Legacy CPG constants remain only for dormant diagnostic/helper functions.
# They are NOT exposed to the policy and are NOT registered as active rewards.
# The active diagonal coupling below is triggered by measured liftoff/touchdown
# events. 45 mm is the current collision-surface lift objective.
AIR_TIME_TARGET_S = 0.16
GAIT_FREQUENCY_HZ = 1.00       # legacy helper default only; inactive in RewardsCfg
GAIT_SWING_RATIO = 0.38        # legacy helper default only; inactive in RewardsCfg
GAIT_PHASE_OFFSET = 0.40       # legacy helper default only; inactive in RewardsCfg
GAIT_STARTUP_DELAY_S = 0.50    # legacy helper default only; inactive in RewardsCfg
GAIT_STRIDE_LENGTH_M = 0.13    # legacy helper default only; inactive in RewardsCfg
# Max obstacle/stair height is 40 mm; use 45 mm collision-surface clearance.
# This is 22.5% of the 200 mm fully extended leg length.
GAIT_CLEARANCE_M = 0.045

# Collision-geometry clearance measurement. The ray pattern queries the static
# terrain mesh around each moving foot. Geometry points are loaded from USD once
# and reduced to convex-support samples before batched GPU evaluation.
# The ray grid is centered at the CAD/IK-Preview-validated sole-center offset
# in each foot rigid-body frame. Actual collision vertices are still extracted
# from the USD at runtime and used for the clearance calculation itself.
FOOT_SOLE_RAY_CENTER_LOCAL_M = (
    (-0.00887, 0.10678, 0.01734),  # foot_1
    (+0.00888, 0.10678, 0.01734),  # foot_2
    (+0.00888, 0.10678, 0.01734),  # foot_3
    (-0.00887, 0.10678, 0.01734),  # foot_4
)
FOOT_CLEARANCE_RAY_GRID_SIZE_M = (0.20, 0.20)
FOOT_CLEARANCE_RAY_RESOLUTION_M = 0.02
FOOT_CLEARANCE_RAY_ORIGIN_HEIGHT_M = 0.15
FOOT_CLEARANCE_RAY_MAX_DISTANCE_M = 1.00
FOOT_CLEARANCE_MAX_NEAREST_RAY_DISTANCE_M = 0.020
FOOT_COLLISION_SUPPORT_DIRECTIONS = 96

# Hysteresis prevents contact chatter around the liftoff/touchdown threshold.
FOOT_CONTACT_ON_THRESHOLD_N = 1.0
FOOT_CONTACT_OFF_THRESHOLD_N = 0.5

# Event synchronization is a soft preference, not a periodic schedule.
DIAGONAL_EVENT_MATCH_WINDOW_S = 0.12
DIAGONAL_EVENT_MATCH_SIGMA_S = 0.04

# Progress reward uses signed root-position displacement each policy step.
# Summed without discounting, the increments equal total net displacement.
# ---------------------------------------------------------------------------
# FEETECH ST-3215-C018 / STS3215 12 V (30 kg.cm) actuator model
# ---------------------------------------------------------------------------
# Manufacturer specification, edition A/0, 2023-07-20:
#   Rated supply: 12 V (operating input range: 4-14 V)
#   No-load speed: 0.222 s/60 deg = 45 rpm = ~270 deg/s
#   Rated torque/current: 10 kg.cm / 0.90 A
#   Stall torque/current: 30 kg.cm / 2.70 A
#   Gear backlash: <= 0.5 deg; gear ratio: 1:345
# Stall torque is a transient limit, NOT continuous operating torque.
# This model uses the rated torque as effort_limit and stall torque only as
# the peak/saturation envelope. Electrical current/thermal dynamics are not
# fully simulated; inferred current should be validated on physical hardware.
STS3215_SUPPLY_V = 12.0
STS3215_OPERATING_VOLTAGE_MIN_V = 4.0
STS3215_OPERATING_VOLTAGE_MAX_V = 14.0
STS3215_RATED_CURRENT_A = 0.90
STS3215_STALL_CURRENT_A = 2.70
STS3215_NO_LOAD_CURRENT_A = 0.18
STS3215_TORQUE_CONSTANT_KGCM_PER_A = 11.0
STS3215_GEAR_RATIO = 345.0
KGFCM_TO_NM = 0.0980665
_STS3215_STALL_KGCM = 30.0
_STS3215_RATED_KGCM = 10.0
_STS3215_NO_LOAD_SECONDS_PER_60_DEG = 0.222
if not (STS3215_OPERATING_VOLTAGE_MIN_V <= STS3215_SUPPLY_V <= STS3215_OPERATING_VOLTAGE_MAX_V):
    raise ValueError("STS3215 12 V servo supply is outside the documented 4-14 V range.")
STS3215_PEAK_TORQUE_NM = _STS3215_STALL_KGCM * KGFCM_TO_NM
STS3215_RATED_TORQUE_NM = _STS3215_RATED_KGCM * KGFCM_TO_NM
STS3215_NO_LOAD_SPEED = math.radians(60.0) / _STS3215_NO_LOAD_SECONDS_PER_60_DEG

# Mechanical dimensions supplied for this RobotDog revision.
ROBOT_BODY_LENGTH_M = 0.25
ROBOT_LEG_MAX_EXTENSION_M = 0.20

SERVO_KP = 18.0
SERVO_KD = 1.0


# =============================================================================
# JOINT TARGET TRAJECTORY LIMITS
# =============================================================================
# Limits the commanded joint trajectory, not the physical joint velocity.
#
# ST-3215-C018 no-load speed is ~270 deg/s at 12 V.
# Keep the commanded reference cap below no-load speed to preserve loaded-motion margin.

# Conservative reference trajectory cap; loaded servo motion is slower than no-load spec.
JOINT_TARGET_MAX_VELOCITY_DEG_S = 150.0  # ~56% of 12 V no-load speed; conservative loaded-motion cap
JOINT_TARGET_MAX_ACCELERATION_DEG_S2 = 900.0

JOINT_TARGET_MAX_VELOCITY_RAD_S = math.radians(
    JOINT_TARGET_MAX_VELOCITY_DEG_S
)

JOINT_TARGET_MAX_ACCELERATION_RAD_S2 = math.radians(
    JOINT_TARGET_MAX_ACCELERATION_DEG_S2
)

# ============================================================================
# SERVO COMMAND POSITION DEADBAND
# ============================================================================
# IMPORTANT: acesta este un DEADBAND AL COMENZII TRIMISE SERVO-ULUI.
# Comparăm fiecare target nou cu ULTIMUL TARGET ACCEPTAT de actuator.
#
# Dacă schimbarea este < 1.5°, noul target NU este transmis actuatorului.
# Actuatorul continuă să primească ultimul target acceptat.
#
# Dacă schimbarea este >= 1.5°, noul target este acceptat și devine noua
# referință pentru următoarea comparație.
SERVO_COMMAND_DEADBAND_DEG = 1.0  # calibration test: no command deadband
SERVO_COMMAND_DEADBAND_RAD = math.radians(SERVO_COMMAND_DEADBAND_DEG)

# =============================================================================
# POSITION ERROR DEADZONE
# =============================================================================
# Deadzone applied to the PHYSICAL POSITION ERROR seen by the actuator.
#
# Example with 0.8 deg deadzone and target = 30 deg:
#   q = 29.3 deg -> effective error = 0.0 deg
#   q = 30.2 deg -> effective error = 0.0 deg
#   q = 28.9 deg -> effective error = -? / +0.3 deg toward target
#
# IMPORTANT:
# This is NOT a command deadband. The accepted target remains unchanged.
# We only remove the small proportional correction around q ~= target,
# which is what breaks the 29.x -> 30.x -> 29.x hunting cycle.
# The transition is continuous: effective_error = sign(error) *
# max(|error| - deadzone, 0).
POSITION_ERROR_DEADBAND_DEG = 0.6
POSITION_ERROR_DEADBAND_RAD = math.radians(POSITION_ERROR_DEADBAND_DEG)

# =============================================================================
# SERVO UNIT-TO-UNIT VARIATION
# =============================================================================

# Independent variation for every servo in every environment.
# We deliberately make the range broad but still physically plausible.
SERVO_TORQUE_SCALE_MIN = 0.90
SERVO_TORQUE_SCALE_MAX = 1.00

SERVO_SPEED_SCALE_MIN = 0.90
SERVO_SPEED_SCALE_MAX = 1.00

# Joint zero/calibration error.
# This models a servo whose internal zero is not perfectly aligned
# with the mechanical joint zero.
SERVO_ZERO_OFFSET_MIN_DEG = 0.0
SERVO_ZERO_OFFSET_MAX_DEG = 0.0

SERVO_ZERO_OFFSET_MIN_RAD = math.radians(SERVO_ZERO_OFFSET_MIN_DEG)
SERVO_ZERO_OFFSET_MAX_RAD = math.radians(SERVO_ZERO_OFFSET_MAX_DEG)




# Actuator command delay: physics steps.
ACTUATOR_DELAY_MIN = 0
ACTUATOR_DELAY_MAX = 1  # 0-5 ms randomized command latency at 200 Hz


# =============================================================================
# JOINT FEEDBACK LATENCY
# =============================================================================

# Feedback latency from servo encoder/electronics/bus to the controller.
#
# Policy runs at 50 Hz:
#   1 step = 20 ms
#   2 steps = 40 ms
#
# Randomized independently per environment at every reset.
JOINT_FEEDBACK_DELAY_MIN_STEPS = 0
JOINT_FEEDBACK_DELAY_MAX_STEPS = 1  # 0-20 ms at 50 Hz



# Control-loop command shaping is implemented inside DelayedDCMotor.
# This includes target trajectory velocity and acceleration limiting.

# Mechanical / joint constraints.
JOINT_LIMITS_DEG = {
    "revolute_1": (-90.0, +90.0),
    "revolute_2": (-90.0, +90.0),
    "revolute_3": (-180.0, +60.0),
    "revolute_4": (-90.0, +90.0),
    "revolute_5": (-90.0, +90.0),
    "revolute_6": (-60.0, +180.0),
    "revolute_7": (-90.0, +90.0),
    "revolute_8": (-90.0, +90.0),
    "revolute_9": (-180.0, +60.0),
    "revolute_10": (-90.0, +90.0),
    "revolute_11": (-90.0, +90.0),
    "revolute_12": (-60, +180.0),
}

JOINT_LIMITS_RAD = {
    name: (math.radians(lo), math.radians(hi))
    for name, (lo, hi) in JOINT_LIMITS_DEG.items()
}

# Standing position from your original config.
DEFAULT_JOINT_POS = {
    "revolute_1": math.radians(0.0),
    "revolute_2": math.radians(+35.0),
    "revolute_3": math.radians(+10.0),
    "revolute_4": math.radians(0.0),
    "revolute_5": math.radians(-35.0),
    "revolute_6": math.radians(-10.0),
    "revolute_7": math.radians(0.0),
    "revolute_8": math.radians(+35.0),
    "revolute_9": math.radians(+10.0),
    "revolute_10": math.radians(0.0),
    "revolute_11": math.radians(-35.0),
    "revolute_12": math.radians(-10.0),
}


# =============================================================================
# REALISTIC IMU STATE
# =============================================================================

def _initialize_imu_randomization_state(env):
    """Create per-environment IMU parameters/state tensors once."""

    if hasattr(env, "_robotdog_imu_state"):
        return env._robotdog_imu_state

    n = env.num_envs
    device = env.device

    state = {
        # ------------------------------------------------------------------
        # Static scale-factor errors
        # Shape: [num_envs, 3]
        # ------------------------------------------------------------------
        "gyro_scale": torch.ones((n, 3), device=device),
        "accel_scale": torch.ones((n, 3), device=device),

        # ------------------------------------------------------------------
        # Current bias
        # ------------------------------------------------------------------
        "gyro_bias": torch.zeros((n, 3), device=device),
        "accel_bias": torch.zeros((n, 3), device=device),

        # ------------------------------------------------------------------
        # Bias random-walk / drift state
        # ------------------------------------------------------------------
        "gyro_bias_drift": torch.zeros((n, 3), device=device),
        "accel_bias_drift": torch.zeros((n, 3), device=device),

        "last_gyro_measurement": torch.zeros(
            (n, 3),
            device=device,
        ),

        "last_accel_measurement": torch.zeros(
            (n, 3),
            device=device,
        ),

        # ------------------------------------------------------------------
        # Mounting / axis misalignment
        # Euler angles: roll, pitch, yaw
        # ------------------------------------------------------------------
        "misalign_rpy": torch.zeros((n, 3), device=device),

        # Rotation matrix used later to transform sensor measurements.
        # [num_envs, 3, 3]
        "misalign_rot": torch.eye(3, device=device).unsqueeze(0).repeat(n, 1, 1),

        # ------------------------------------------------------------------
        # IMU position offset relative to nominal sensor location
        # ------------------------------------------------------------------
        "position_offset": torch.zeros((n, 3), device=device),

        # ------------------------------------------------------------------
        # Sensor latency
        # ------------------------------------------------------------------
        "delay_steps": torch.zeros(
            (n,),
            dtype=torch.long,
            device=device,
        ),

        # ------------------------------------------------------------------
        # Measurement saturation
        # ------------------------------------------------------------------
        "gyro_saturation": torch.full(
            (n, 3),
            IMU_GYRO_SATURATION_MAX,
            device=device,
        ),

        "accel_saturation": torch.full(
            (n, 3),
            IMU_ACCEL_SATURATION_MAX,
            device=device,
        ),
    }

    # Delay buffers are created once and randomized per environment later.
    state["gyro_delay_buffer"] = DelayBuffer(
        IMU_DELAY_MAX_STEPS,
        n,
        device=device,
    )

    state["accel_delay_buffer"] = DelayBuffer(
        IMU_DELAY_MAX_STEPS,
        n,
        device=device,
    )

    env._robotdog_imu_state = state

    return state


def _sample_imu_randomization(
    env,
    env_ids: Sequence[int] | None = None,
):
    """
    Randomize the complete static IMU characteristics for selected environments.

    This is called at reset, so every episode can receive a different
    virtual IMU personality.
    """

    state = _initialize_imu_randomization_state(env)

    if env_ids is None or env_ids == slice(None):
        active_env_ids = slice(None)
        num_envs = env.num_envs
    else:
        active_env_ids = env_ids
        num_envs = len(env_ids)

    device = env.device

    # ----------------------------------------------------------------------
    # SCALE FACTORS
    # ----------------------------------------------------------------------

    gyro_scale = torch.empty(
        (num_envs, 3),
        device=device,
    ).uniform_(
        IMU_GYRO_SCALE_MIN,
        IMU_GYRO_SCALE_MAX,
    )

    accel_scale = torch.empty(
        (num_envs, 3),
        device=device,
    ).uniform_(
        IMU_ACCEL_SCALE_MIN,
        IMU_ACCEL_SCALE_MAX,
    )

    # ----------------------------------------------------------------------
    # INITIAL BIAS
    # ----------------------------------------------------------------------

    gyro_bias = torch.empty(
        (num_envs, 3),
        device=device,
    ).normal_(
        mean=0.0,
        std=IMU_GYRO_BIAS_STD,
    )

    accel_bias = torch.empty(
        (num_envs, 3),
        device=device,
    ).normal_(
        mean=0.0,
        std=IMU_ACCEL_BIAS_STD,
    )

    # ----------------------------------------------------------------------
    # RESET DRIFT STATE
    # ----------------------------------------------------------------------

    gyro_bias_drift = torch.zeros(
        (num_envs, 3),
        device=device,
    )

    accel_bias_drift = torch.zeros(
        (num_envs, 3),
        device=device,
    )

    # ----------------------------------------------------------------------
    # MOUNTING MISALIGNMENT
    # ----------------------------------------------------------------------

    roll = torch.empty(
        (num_envs,),
        device=device,
    ).uniform_(
        IMU_MISALIGN_ROLL_MIN_RAD,
        IMU_MISALIGN_ROLL_MAX_RAD,
    )

    pitch = torch.empty(
        (num_envs,),
        device=device,
    ).uniform_(
        IMU_MISALIGN_PITCH_MIN_RAD,
        IMU_MISALIGN_PITCH_MAX_RAD,
    )

    yaw = torch.empty(
        (num_envs,),
        device=device,
    ).uniform_(
        IMU_MISALIGN_YAW_MIN_RAD,
        IMU_MISALIGN_YAW_MAX_RAD,
    )

    misalign_rpy = torch.stack(
        [roll, pitch, yaw],
        dim=-1,
    )
    
    misalign_rot = _build_imu_misalignment_rotation(
        roll,
        pitch,
        yaw,
    )

    # ----------------------------------------------------------------------
    # POSITION OFFSET
    # ----------------------------------------------------------------------

    position_offset = torch.empty(
        (num_envs, 3),
        device=device,
    ).uniform_(
        IMU_POSITION_OFFSET_MIN_M,
        IMU_POSITION_OFFSET_MAX_M,
    )

    # ----------------------------------------------------------------------
    # SENSOR LATENCY
    # ----------------------------------------------------------------------

    delay_steps = torch.randint(
        low=IMU_DELAY_MIN_STEPS,
        high=IMU_DELAY_MAX_STEPS + 1,
        size=(num_envs,),
        dtype=torch.long,
        device=device,
    )

    # ----------------------------------------------------------------------
    # SATURATION
    # Randomized independently for X/Y/Z.
    # ----------------------------------------------------------------------

    gyro_saturation = torch.empty(
        (num_envs, 3),
        device=device,
    ).uniform_(
        IMU_GYRO_SATURATION_MIN,
        IMU_GYRO_SATURATION_MAX,
    )

    accel_saturation = torch.empty(
        (num_envs, 3),
        device=device,
    ).uniform_(
        IMU_ACCEL_SATURATION_MIN,
        IMU_ACCEL_SATURATION_MAX,
    )

    # ----------------------------------------------------------------------
    # STORE
    # ----------------------------------------------------------------------

    state["gyro_scale"][active_env_ids] = gyro_scale
    state["accel_scale"][active_env_ids] = accel_scale

    state["gyro_bias"][active_env_ids] = gyro_bias
    state["accel_bias"][active_env_ids] = accel_bias

    state["gyro_bias_drift"][active_env_ids] = gyro_bias_drift
    state["accel_bias_drift"][active_env_ids] = accel_bias_drift

    state["misalign_rpy"][active_env_ids] = misalign_rpy
    state["misalign_rot"][active_env_ids] = misalign_rot
    state["position_offset"][active_env_ids] = position_offset

    state["delay_steps"][active_env_ids] = delay_steps

    state["gyro_saturation"][active_env_ids] = gyro_saturation
    state["accel_saturation"][active_env_ids] = accel_saturation

    # Delay buffers must know the delay of each selected environment.
    state["gyro_delay_buffer"].set_time_lag(
        delay_steps.to(dtype=torch.int),
        env_ids,
    )

    state["accel_delay_buffer"].set_time_lag(
        delay_steps.to(dtype=torch.int),
        env_ids,
    )

    state["gyro_delay_buffer"].reset(env_ids)
    state["accel_delay_buffer"].reset(env_ids)

# =============================================================================
# JOINT FEEDBACK DELAY
# =============================================================================

def _initialize_joint_feedback_delay_state(env):
    """
    Create the joint-feedback latency state once.

    Position and velocity use separate delay buffers, but share
    the same latency per environment because they would normally
    arrive as part of the same servo feedback update.
    """

    if hasattr(env, "_robotdog_joint_feedback_state"):
        return env._robotdog_joint_feedback_state

    n = env.num_envs
    device = env.device

    state = {
        "delay_steps": torch.zeros(
            (n,),
            dtype=torch.long,
            device=device,
        ),

        "position_delay_buffer": DelayBuffer(
            JOINT_FEEDBACK_DELAY_MAX_STEPS,
            n,
            device=device,
        ),

        "velocity_delay_buffer": DelayBuffer(
            JOINT_FEEDBACK_DELAY_MAX_STEPS,
            n,
            device=device,
        ),
    }

    env._robotdog_joint_feedback_state = state

    return state


def _sample_joint_feedback_delay(
    env,
    env_ids: Sequence[int] | None = None,
):
    """
    Randomize servo joint-feedback latency at reset.

    One latency value is sampled per environment and is shared between
    joint position and joint velocity feedback.
    """

    state = _initialize_joint_feedback_delay_state(env)

    if env_ids is None or env_ids == slice(None):
        active_env_ids = slice(None)
        num_envs = env.num_envs
    else:
        active_env_ids = env_ids
        num_envs = len(env_ids)

    delay_steps = torch.randint(
        low=JOINT_FEEDBACK_DELAY_MIN_STEPS,
        high=JOINT_FEEDBACK_DELAY_MAX_STEPS + 1,
        size=(num_envs,),
        dtype=torch.long,
        device=env.device,
    )

    state["delay_steps"][active_env_ids] = delay_steps

    # Same latency for position and velocity because both belong
    # to the same servo feedback stream.
    state["position_delay_buffer"].set_time_lag(
        delay_steps.to(dtype=torch.int),
        env_ids,
    )

    state["velocity_delay_buffer"].set_time_lag(
        delay_steps.to(dtype=torch.int),
        env_ids,
    )

    # Clear stale history after changing the latency.
    state["position_delay_buffer"].reset(env_ids)
    state["velocity_delay_buffer"].reset(env_ids)


def _joint_pos_feedback_delayed(env):
    """
    Joint-position feedback as seen by the controller.

    The actual simulated joint position is delayed before the
    existing observation noise/bias model is applied.
    """

    state = _initialize_joint_feedback_delay_state(env)

    joint_pos = mdp.joint_pos_rel(env)

    return state["position_delay_buffer"].compute(joint_pos)


def _joint_vel_feedback_delayed(env):
    """
    Joint-velocity feedback as seen by the controller.

    The actual simulated joint velocity is delayed before the
    existing observation noise/bias model is applied.
    """

    state = _initialize_joint_feedback_delay_state(env)

    joint_vel = mdp.joint_vel_rel(env)

    return state["velocity_delay_buffer"].compute(joint_vel)

# =============================================================================
# IMU MISALIGNMENT ROTATION
# =============================================================================

def _build_imu_misalignment_rotation(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """
    Build per-environment XYZ rotation matrices for IMU mounting error.

    The rotations are applied in Z-Y-X order:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Input shapes:
        roll  -> [N]
        pitch -> [N]
        yaw   -> [N]

    Output:
        R -> [N, 3, 3]
    """

    cx = torch.cos(roll)
    sx = torch.sin(roll)

    cy = torch.cos(pitch)
    sy = torch.sin(pitch)

    cz = torch.cos(yaw)
    sz = torch.sin(yaw)

    zeros = torch.zeros_like(roll)
    ones = torch.ones_like(roll)

    rx = torch.stack(
        [
            ones, zeros, zeros,
            zeros, cx, -sx,
            zeros, sx, cx,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    ry = torch.stack(
        [
            cy, zeros, sy,
            zeros, ones, zeros,
            -sy, zeros, cy,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    rz = torch.stack(
        [
            cz, -sz, zeros,
            sz, cz, zeros,
            zeros, zeros, ones,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    return rz @ ry @ rx


# =============================================================================
# IMU MEASUREMENT MODEL - BIAS DRIFT + SCALE FACTOR
# =============================================================================

def _update_imu_bias_drift(env):
    """Advance the per-environment IMU bias random walk."""

    state = _initialize_imu_randomization_state(env)

    # One policy observation arrives every DECIMATION physics steps.
    # We therefore accumulate the random walk over the control period.
    dt = PHYSICS_DT * DECIMATION

    gyro_drift_noise = torch.randn_like(
        state["gyro_bias_drift"]
    )

    accel_drift_noise = torch.randn_like(
        state["accel_bias_drift"]
    )

    state["gyro_bias_drift"] += (
        IMU_GYRO_BIAS_DRIFT_STD
        * math.sqrt(dt)
        * gyro_drift_noise
    )

    state["accel_bias_drift"] += (
        IMU_ACCEL_BIAS_DRIFT_STD
        * math.sqrt(dt)
        * accel_drift_noise
    )


def _imu_ang_vel_realistic(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
):
    """
    Realistic gyro measurement:
        scale factor
        + static bias
        + slowly drifting bias
    """

    state = _initialize_imu_randomization_state(env)

    # Advance slowly-varying sensor bias.
    _update_imu_bias_drift(env)

    

    true_ang_vel = env.scene["imu"].data.ang_vel_b

    # Transform the angular velocity through the randomized
    # IMU mounting / axis misalignment.
    misaligned_ang_vel = torch.bmm(
        state["misalign_rot"],
        true_ang_vel.unsqueeze(-1),
    ).squeeze(-1)

    measured_ang_vel = (
        misaligned_ang_vel
        * state["gyro_scale"]
        + state["gyro_bias"]
        + state["gyro_bias_drift"]
    )

    # Electronic saturation.
    measured_ang_vel = torch.clamp(
        measured_ang_vel,
        min=-state["gyro_saturation"],
        max=state["gyro_saturation"],
    )

    # Sensor latency at policy-observation rate.
    measured_ang_vel = state["gyro_delay_buffer"].compute(
        measured_ang_vel
    )

    state["last_gyro_measurement"] = measured_ang_vel

    return measured_ang_vel


def _imu_lin_acc_realistic(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
):
    """
    Realistic accelerometer measurement:
        scale factor
        + static bias
        + slowly drifting bias
    """

    state = _initialize_imu_randomization_state(env)

    true_lin_acc = env.scene["imu"].data.lin_acc_b

    # ----------------------------------------------------------------------
    # POSITION OFFSET EFFECT
    # ----------------------------------------------------------------------
    # The accelerometer at an offset from the base origin experiences:
    #
    #   alpha x r + omega x (omega x r)
    #
    # where:
    #   alpha = angular acceleration
    #   omega = angular velocity
    #   r     = IMU position offset.
    #
    # This makes the simulated accelerometer react to rotational motion
    # exactly like a sensor mounted away from the rotation center.

    robot = env.scene["robot"]

    omega = robot.data.root_ang_vel_b

    # Angular acceleration of the root body.
    # Isaac Lab 2.0.x does not expose root_ang_acc_b directly.

    alpha_w = robot.data.body_ang_acc_w[:, 0, :]

    # Convert angular acceleration from world frame to base frame.
    alpha = math_utils.quat_apply_inverse(robot.data.root_quat_w, alpha_w)

    r = state["position_offset"]

    tangential_acc = torch.cross(
        alpha,
        r,
        dim=-1,
    )

    centripetal_acc = torch.cross(
        omega,
        torch.cross(
            omega,
            r,
            dim=-1,
        ),
        dim=-1,
    )

    offset_acc = tangential_acc + centripetal_acc

    # Add acceleration caused by the physical IMU mounting position.
    offset_lin_acc = true_lin_acc + offset_acc

    # ----------------------------------------------------------------------
    # AXIS MISALIGNMENT
    # ----------------------------------------------------------------------

    misaligned_lin_acc = torch.bmm(
        state["misalign_rot"],
        offset_lin_acc.unsqueeze(-1),
    ).squeeze(-1)

    # ----------------------------------------------------------------------
    # SCALE FACTOR + BIAS + BIAS DRIFT
    # ----------------------------------------------------------------------

    measured_lin_acc = (
        misaligned_lin_acc
        * state["accel_scale"]
        + state["accel_bias"]
        + state["accel_bias_drift"]
    )

    # Electronic saturation.
    measured_lin_acc = torch.clamp(
        measured_lin_acc,
        min=-state["accel_saturation"],
        max=state["accel_saturation"],
    )

    # Sensor latency at policy-observation rate.
    measured_lin_acc = state["accel_delay_buffer"].compute(
        measured_lin_acc
    )

    state["last_accel_measurement"] = measured_lin_acc

    return measured_lin_acc

# =============================================================================
# IMU ATTITUDE / PROJECTED GRAVITY ESTIMATOR
# =============================================================================

ATTITUDE_ESTIMATOR_GRAVITY_MPS2 = 9.81

# Maximum normalized acceleration deviation over which
# accelerometer correction is gradually trusted.
ATTITUDE_ESTIMATOR_ACCEL_TRUST_WIDTH = 0.35

# Complementary-filter correction strength per policy step.
ATTITUDE_ESTIMATOR_CORRECTION_GAIN = 0.08


def _initialize_attitude_estimator_state(env):
    """Create the per-environment gravity-direction estimator state."""

    if hasattr(env, "_robotdog_attitude_estimator_state"):
        return env._robotdog_attitude_estimator_state

    n = env.num_envs
    device = env.device

    state = {
        # Estimated gravity direction expressed in IMU/body frame.
        # At rest this should be approximately [0, 0, -1].
        "gravity_b": torch.zeros(
            (n, 3),
            device=device,
        ),

        "initialized": torch.zeros(
            n,
            dtype=torch.bool,
            device=device,
        ),
    }

    env._robotdog_attitude_estimator_state = state

    return state


def estimated_projected_gravity(env):
    """
    Estimate projected gravity from the realistic IMU gyro + accelerometer.

    This is a complementary filter:
      1. gyro propagates the gravity vector
      2. accelerometer slowly corrects roll/pitch drift
      3. yaw is intentionally not estimated

    Output:
        normalized gravity direction in body/IMU frame
    """

    estimator = _initialize_attitude_estimator_state(env)
    imu_state = _initialize_imu_randomization_state(env)

    gyro = imu_state["last_gyro_measurement"]
    accel = imu_state["last_accel_measurement"]

    accel_norm = torch.linalg.norm(
        accel,
        dim=-1,
        keepdim=True,
    )

    valid_accel = accel_norm > 1.0e-6

    # Proper acceleration points upward at rest.
    # Gravity points downward.
    measured_gravity = -accel / accel_norm.clamp_min(1.0e-6)

    # ------------------------------------------------------------------
    # GYRO PROPAGATION
    # ------------------------------------------------------------------

    previous_gravity = estimator["gravity_b"]

    propagated_gravity = (
        previous_gravity
        - torch.cross(
            gyro,
            previous_gravity,
            dim=-1,
        ) * POLICY_DT
    )

    propagated_gravity = propagated_gravity / (
        torch.linalg.norm(
            propagated_gravity,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-6)
    )

    # ------------------------------------------------------------------
    # ACCELEROMETER TRUST
    # ------------------------------------------------------------------
    # When |a| is close to 1 g, accelerometer gives useful gravity direction.
    # During strong linear acceleration, trust it less.

    normalized_accel_norm = (
        accel_norm / ATTITUDE_ESTIMATOR_GRAVITY_MPS2
    )

    accel_deviation = torch.abs(
        normalized_accel_norm - 1.0
    )

    accel_trust = torch.clamp(
        1.0
        - accel_deviation
        / ATTITUDE_ESTIMATOR_ACCEL_TRUST_WIDTH,
        min=0.0,
        max=1.0,
    )

    correction_gain = (
        ATTITUDE_ESTIMATOR_CORRECTION_GAIN
        * accel_trust
    )

    corrected_gravity = (
        (1.0 - correction_gain) * propagated_gravity
        + correction_gain * measured_gravity
    )

    corrected_gravity = corrected_gravity / (
        torch.linalg.norm(
            corrected_gravity,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-6)
    )

    # ------------------------------------------------------------------
    # FIRST SAMPLE AFTER RESET
    # ------------------------------------------------------------------

    default_gravity = torch.zeros_like(corrected_gravity)
    default_gravity[:, 2] = -1.0

    corrected_gravity = torch.where(
        valid_accel,
        corrected_gravity,
        default_gravity,
    )

    first_sample = ~estimator["initialized"]

    corrected_gravity = torch.where(
        first_sample.unsqueeze(-1),
        torch.where(
            valid_accel,
            measured_gravity,
            default_gravity,
        ),
        corrected_gravity,
    )

    estimator["gravity_b"] = corrected_gravity

    estimator["initialized"] = (
        estimator["initialized"]
        | first_sample
    )

    return corrected_gravity


# =============================================================================
# CUSTOM REWARDS
# =============================================================================
#
# Toată această secțiune folosește ground truth din simulator.
# Politica NU vede aceste semnale.
#
# Avantaj:
#   putem pedepsi slip / impact / limite / instabilitate fără să-i dăm
#   robotului informații pe care hardware-ul real nu le are.
# =============================================================================





def commanded_progress_reward(
    env,
    command_name: str = "base_velocity",
):
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    command_xy = command[:, :2]
    command_speed = torch.linalg.norm(command_xy, dim=-1)

    moving_command = command_speed > 0.10

    command_dir = command_xy / command_speed.clamp_min(1.0e-6).unsqueeze(-1)

    velocity_xy = robot.data.root_lin_vel_b[:, :2]

    projected_velocity = torch.sum(
        velocity_xy * command_dir,
        dim=-1,
    )

    # Doar progresul în direcția comandată contează.
    forward_progress = torch.relu(projected_velocity)

    # Normalizează astfel încât reward-ul să crească odată cu
    # apropierea de viteza comandată.
    progress_ratio = (
        forward_progress
        / command_speed.clamp_min(0.10)
    )

    return torch.clamp(progress_ratio, 0.0, 1.5) * moving_command.float()


def base_contact_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    force_scale: float = 40.0,
):
    """Bounded penalty when the base hits terrain, to discourage falling forward."""
    sensor = env.scene[sensor_cfg.name]
    force_norm = torch.linalg.norm(sensor.data.net_forces_w, dim=-1)
    max_force = force_norm.amax(dim=-1)
    return torch.clamp(
        (max_force - threshold) / max(force_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )


def base_contact_termination(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
):
    # ---------------------------------------------------------
    # BASE CONTACT
    # ---------------------------------------------------------

    base_sensor = env.scene[sensor_cfg.name]

    base_force = torch.linalg.norm(
        base_sensor.data.net_forces_w,
        dim=-1,
    )

    base_hit = torch.any(
        base_force > threshold,
        dim=-1,
    )

    

    # ---------------------------------------------------------
    # EPISODE TERMINATES IF EITHER HAPPENS
    # ---------------------------------------------------------

    return base_hit 
        
FOOT_BODY_NAMES = [
    "componentfinal_1__body1",
    "componentfinal_mirror__1__body1",
    "componentfinal_mirror___1__1__body1",
    "componentfinal__1__1__body1",
]

SERVO3_BODY_NAMES = [
    "servo_3_1__body1",
    "servo_3_mirror__1__body1",
    "servo_3_mirror___1__1__body1",
    "servo_3__1__1__body1",
]


FOOT_SENSOR_NAMES = ("foot_1", "foot_2", "foot_3", "foot_4")
SERVO3_SENSOR_NAMES = (
    "servo3_contact_1",
    "servo3_contact_2",
    "servo3_contact_3",
    "servo3_contact_4",
)
EXPECTED_CONTACT_SENSOR_BODY_NAMES = {
    **dict(zip(FOOT_SENSOR_NAMES, FOOT_BODY_NAMES)),
    **dict(zip(SERVO3_SENSOR_NAMES, SERVO3_BODY_NAMES)),
}
GROUND_CONTACT_FORCE_THRESHOLD_N = 1.0


def _validate_contact_sensor_source_body(sensor, sensor_name: str) -> None:
    """Fail fast if a contact sensor is not initialized/mapped to its intended source and terrain."""
    if hasattr(sensor, "is_initialized") and not bool(sensor.is_initialized):
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' is not initialized. Do not interpret its zero buffers as no-contact data."
        )

    # The force matrix is one-to-many: one sole/servo3 source body against one
    # or more terrain bodies. Reject a missing/wrong filter before reading zeros.
    sensor_cfg = getattr(sensor, "cfg", None)
    filter_exprs = getattr(sensor_cfg, "filter_prim_paths_expr", None)
    if filter_exprs is not None:
        normalized_filters = [str(expr).rstrip("/") for expr in filter_exprs]
        if "/World/ground/terrain/mesh" not in normalized_filters:
            raise RuntimeError(
                f"Contact sensor '{sensor_name}' is not filtered against '/World/ground/terrain/mesh'. "
                f"Configured filters: {normalized_filters}."
            )

    num_bodies = getattr(sensor, "num_bodies", None)
    if num_bodies is not None and int(num_bodies) != 1:
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' resolved {num_bodies} source bodies; "
            "the PhysX filtered force matrix in Isaac Lab 2.3.1 requires exactly one source body per sensor."
        )

    expected_body_name = EXPECTED_CONTACT_SENSOR_BODY_NAMES.get(sensor_name)
    actual_body_names = getattr(sensor, "body_names", None)
    if actual_body_names is None or expected_body_name is None:
        return

    # Isaac Lab v2.3.1 returns leaf names; retain path normalization for backends
    # that may expose full USD paths.
    actual_leaf_names = [str(name).rstrip("/").split("/")[-1] for name in actual_body_names]
    if actual_leaf_names != [expected_body_name]:
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' resolved source bodies {actual_body_names}; "
            f"expected exactly one body ending in '{expected_body_name}'. Verify its USD prim_path."
        )


def _robotdog_current_step_key(env) -> int:
    """Return a stable key for all observations/rewards computed after one env step."""
    step_counter = getattr(env, "common_step_counter", None)
    if step_counter is not None:
        return int(step_counter)
    return int(env.episode_length_buf.max().item())


def _filtered_terrain_force_data(env, sensor_name: str):
    """Return (filtered force magnitude [N], data-valid mask [N], finite fraction).

    ContactSensor force_matrix_w is [env, sensor-body, filtered-body, xyz].
    The filtered PhysX view is only well-defined for exactly one sensor body
    per environment. NaN/non-finite entries are treated as missing samples,
    never as contact and never allowed to propagate into a reward.
    """
    step_key = _robotdog_current_step_key(env)
    cache = getattr(env, "_robotdog_filtered_force_data_cache", None)
    if cache is not None and cache["step_key"] == step_key:
        cached = cache["data"].get(sensor_name)
        if cached is not None:
            return cached

    sensor = env.scene[sensor_name]
    _validate_contact_sensor_source_body(sensor, sensor_name)
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None:
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' returned force_matrix_w=None. "
            "Check activate_contact_sensors, prim_path, and filter_prim_paths_expr."
        )
    if (
        force_matrix.ndim != 4
        or force_matrix.shape[0] != env.num_envs
        or force_matrix.shape[-1] != 3
    ):
        raise RuntimeError(
            f"Unexpected force_matrix_w shape for '{sensor_name}': "
            f"{tuple(force_matrix.shape)}; expected [N, 1, M, 3]."
        )
    if force_matrix.shape[1] != 1:
        raise RuntimeError(
            f"Filtered contact sensor '{sensor_name}' resolved {force_matrix.shape[1]} "
            "sensor bodies per environment. PhysX filtered contact reporting requires "
            "exactly one source body per environment. Fix prim_path."
        )
    if force_matrix.shape[2] < 1:
        raise RuntimeError(
            f"Sensor '{sensor_name}' matched zero terrain bodies. The filter "
            "'/World/ground/terrain/mesh' did not resolve to a collision body."
        )

    finite = torch.isfinite(force_matrix)
    finite_per_env = finite.any(dim=(1, 2, 3))
    finite_fraction = finite.float().mean()
    if not finite_per_env.any():
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' returned no finite filtered-force values "
            "for any environment. This is invalid sensor data, not evidence that the "
            "foot is airborne. Check contact reporter activation and the terrain filter."
        )

    # Missing / non-finite entries are zeroed only for arithmetic. `finite_per_env`
    # remains separate so callers can suppress rewards when data are unavailable.
    safe_matrix = torch.nan_to_num(force_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    per_pair_magnitude = torch.linalg.vector_norm(safe_matrix, dim=-1)
    per_body_magnitude = per_pair_magnitude.sum(dim=-1)
    force_magnitude = per_body_magnitude.amax(dim=-1)
    force_magnitude = torch.where(
        finite_per_env, force_magnitude, torch.zeros_like(force_magnitude)
    )
    result = (force_magnitude, finite_per_env, finite_fraction)
    cache = getattr(env, "_robotdog_filtered_force_data_cache", None)
    if cache is None or cache["step_key"] != step_key:
        cache = {"step_key": step_key, "data": {}}
        env._robotdog_filtered_force_data_cache = cache
    cache["data"][sensor_name] = result
    return result


def _filtered_terrain_force_magnitude(env, sensor_name: str) -> torch.Tensor:
    """Per-environment force magnitude against filter-matched terrain only."""
    force, _, _ = _filtered_terrain_force_data(env, sensor_name)
    return force


def _filtered_terrain_contact_mask(
    env,
    sensor_name: str,
    threshold_n: float = GROUND_CONTACT_FORCE_THRESHOLD_N,
) -> torch.Tensor:
    """Ground-only contact mask; invalid sensor data is explicitly NOT contact."""
    if threshold_n < 0.0:
        raise ValueError("threshold_n must be non-negative")
    force, data_valid, _ = _filtered_terrain_force_data(env, sensor_name)
    return (force > threshold_n) & data_valid


def _filtered_terrain_contact_history(
    env,
    sensor_name: str,
):
    """Return chronological terrain-filtered force magnitudes and validity [N,T]."""
    sensor = env.scene[sensor_name]
    _validate_contact_sensor_source_body(sensor, sensor_name)
    force_history = sensor.data.force_matrix_w_history
    if force_history is None:
        raise RuntimeError(
            f"Sensor '{sensor_name}' returned force_matrix_w_history=None. "
            "Check history_length and filter_prim_paths_expr."
        )
    if (
        force_history.ndim != 5
        or force_history.shape[0] != env.num_envs
        or force_history.shape[-1] != 3
    ):
        raise RuntimeError(
            f"Unexpected force_matrix_w_history shape for '{sensor_name}': "
            f"{tuple(force_history.shape)}; expected [N,T,1,M,3]."
        )
    if force_history.shape[2] != 1:
        raise RuntimeError(
            f"Filtered history for '{sensor_name}' resolved {force_history.shape[2]} "
            "sensor bodies; expected exactly one. Fix prim_path."
        )
    if force_history.shape[3] < 1:
        raise RuntimeError(
            f"Sensor '{sensor_name}' has zero filtered terrain bodies in its history."
        )

    finite = torch.isfinite(force_history)
    sample_valid = finite.any(dim=(2, 3, 4))  # [N,T]
    if not sample_valid.any():
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' returned no finite filtered-force history "
            "for any environment/time sample. The history is invalid, not airborne."
        )
    safe_history = torch.nan_to_num(force_history, nan=0.0, posinf=0.0, neginf=0.0)
    per_pair_magnitude = torch.linalg.vector_norm(safe_history, dim=-1)
    per_body_magnitude = per_pair_magnitude.sum(dim=-1)
    force_per_step = per_body_magnitude.amax(dim=-1)  # [N,T]

    # Isaac Lab 2.3.1 stores newest sample at index 0; process oldest -> newest.
    return force_per_step.flip(dims=(1,)), sample_valid.flip(dims=(1,))


def _robotdog_filtered_foot_contact_timing(env) -> dict[str, torch.Tensor]:
    """Integrate terrain-filtered contact transitions at physics-step resolution.

    ``liftoff`` and ``touchdown`` are event pulses for the current policy step.
    The event-age tensors are measured from those physical transitions; they do
    not advance a global gait phase or prescribe when the next transition occurs.
    Invalid sensor samples invalidate event ages rather than being interpreted
    as contact or flight.
    """
    step_key = _robotdog_current_step_key(env)
    cached_timing = getattr(env, "_robotdog_filtered_foot_timing_cache", None)
    if cached_timing is not None and cached_timing["step_key"] == step_key:
        return cached_timing["result"]

    force_histories = []
    valid_histories = []
    for sensor_name in FOOT_SENSOR_NAMES:
        forces_i, valid_i = _filtered_terrain_contact_history(env, sensor_name)
        force_histories.append(forces_i)
        valid_histories.append(valid_i)

    lengths = {int(item.shape[1]) for item in force_histories}
    valid_lengths = {int(item.shape[1]) for item in valid_histories}
    if len(lengths) != 1 or len(valid_lengths) != 1:
        raise RuntimeError(
            f"Inconsistent foot contact-history lengths: contacts={lengths}, validity={valid_lengths}"
        )
    force_history = torch.stack(force_histories, dim=-1)  # [N,T,4]
    valid_history = torch.stack(valid_histories, dim=-1)  # [N,T,4]
    if force_history.shape[1] < 1:
        raise RuntimeError("Foot contact history is empty")

    episode_step = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    n = env.num_envs
    dt = float(PHYSICS_DT)
    inf_age = float("inf")
    if not hasattr(env, "_robotdog_filtered_foot_timing_state"):
        env._robotdog_filtered_foot_timing_state = {
            "previous_contact": torch.zeros((n, 4), dtype=torch.bool, device=env.device),
            "current_air_time": torch.zeros((n, 4), device=env.device),
            "current_contact_time": torch.zeros((n, 4), device=env.device),
            "last_air_time": torch.zeros((n, 4), device=env.device),
            "last_contact_time": torch.zeros((n, 4), device=env.device),
            "touchdown": torch.zeros((n, 4), dtype=torch.bool, device=env.device),
            "liftoff": torch.zeros((n, 4), dtype=torch.bool, device=env.device),
            "time_since_liftoff": torch.full((n, 4), inf_age, device=env.device),
            "time_since_touchdown": torch.full((n, 4), inf_age, device=env.device),
            "initialized": torch.zeros((n, 4), dtype=torch.bool, device=env.device),
            "last_episode_step": episode_step.clone(),
        }

    state = env._robotdog_filtered_foot_timing_state
    reset_mask = episode_step < state["last_episode_step"]
    if torch.any(reset_mask):
        state["previous_contact"][reset_mask] = False
        state["current_air_time"][reset_mask] = 0.0
        state["current_contact_time"][reset_mask] = 0.0
        state["last_air_time"][reset_mask] = 0.0
        state["last_contact_time"][reset_mask] = 0.0
        state["touchdown"][reset_mask] = False
        state["liftoff"][reset_mask] = False
        state["time_since_liftoff"][reset_mask] = inf_age
        state["time_since_touchdown"][reset_mask] = inf_age
        state["initialized"][reset_mask] = False

    advanced = episode_step > state["last_episode_step"]
    first_call = ~state["initialized"].any(dim=-1)
    init_env_mask = reset_mask | first_call

    # Seed from the newest valid sample after reset. Do not infer a transition
    # from stale pre-reset sensor history.
    newest_force = force_history[:, -1, :]
    newest_valid = valid_history[:, -1, :]
    newest_contact = newest_force >= FOOT_CONTACT_ON_THRESHOLD_N
    seed_mask = init_env_mask.unsqueeze(-1) & newest_valid
    state["previous_contact"] = torch.where(
        seed_mask, newest_contact, state["previous_contact"]
    )
    for key in ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time"):
        state[key] = torch.where(seed_mask, torch.zeros_like(state[key]), state[key])
    state["time_since_liftoff"] = torch.where(
        seed_mask, torch.full_like(state["time_since_liftoff"], inf_age), state["time_since_liftoff"]
    )
    state["time_since_touchdown"] = torch.where(
        seed_mask, torch.full_like(state["time_since_touchdown"], inf_age), state["time_since_touchdown"]
    )
    state["touchdown"] = torch.where(
        init_env_mask.unsqueeze(-1), torch.zeros_like(state["touchdown"]), state["touchdown"]
    )
    state["liftoff"] = torch.where(
        init_env_mask.unsqueeze(-1), torch.zeros_like(state["liftoff"]), state["liftoff"]
    )
    state["initialized"] |= seed_mask

    # Consume each new DECIMATION-sized history window exactly once, oldest to newest.
    process_env = advanced & (~init_env_mask)
    if torch.any(process_env):
        state["touchdown"] = torch.where(
            process_env.unsqueeze(-1), torch.zeros_like(state["touchdown"]), state["touchdown"]
        )
        state["liftoff"] = torch.where(
            process_env.unsqueeze(-1), torch.zeros_like(state["liftoff"]), state["liftoff"]
        )
        for history_index in range(force_history.shape[1]):
            sample_force = force_history[:, history_index, :]
            sample_valid = valid_history[:, history_index, :]

            uninitialized_valid = process_env.unsqueeze(-1) & (~state["initialized"]) & sample_valid
            # Contact engages at the ON threshold. Once engaged, it remains
            # engaged until force drops below the lower OFF threshold.
            hysteresis_threshold = torch.where(
                state["previous_contact"],
                torch.full_like(sample_force, FOOT_CONTACT_OFF_THRESHOLD_N),
                torch.full_like(sample_force, FOOT_CONTACT_ON_THRESHOLD_N),
            )
            sample_contact = sample_force >= hysteresis_threshold
            sample_contact = torch.where(
                uninitialized_valid,
                sample_force >= FOOT_CONTACT_ON_THRESHOLD_N,
                sample_contact,
            )
            state["previous_contact"] = torch.where(
                uninitialized_valid, sample_contact, state["previous_contact"]
            )
            for key in ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time"):
                state[key] = torch.where(uninitialized_valid, torch.zeros_like(state[key]), state[key])
            state["time_since_liftoff"] = torch.where(
                uninitialized_valid,
                torch.full_like(state["time_since_liftoff"], inf_age),
                state["time_since_liftoff"],
            )
            state["time_since_touchdown"] = torch.where(
                uninitialized_valid,
                torch.full_like(state["time_since_touchdown"], inf_age),
                state["time_since_touchdown"],
            )
            state["initialized"] |= uninitialized_valid

            active = process_env.unsqueeze(-1) & sample_valid & state["initialized"] & (~uninitialized_valid)
            invalid_sample = process_env.unsqueeze(-1) & (~sample_valid)
            # A gap in the force stream breaks an event pair; never match against
            # stale timestamps retained from before the invalid interval.
            state["time_since_liftoff"] = torch.where(
                invalid_sample,
                torch.full_like(state["time_since_liftoff"], inf_age),
                state["time_since_liftoff"],
            )
            state["time_since_touchdown"] = torch.where(
                invalid_sample,
                torch.full_like(state["time_since_touchdown"], inf_age),
                state["time_since_touchdown"],
            )

            new_touchdown = sample_contact & (~state["previous_contact"])
            new_liftoff = (~sample_contact) & state["previous_contact"]

            old_air = state["current_air_time"]
            old_contact = state["current_contact_time"]
            new_last_air = torch.where(new_touchdown, old_air, state["last_air_time"])
            new_last_contact = torch.where(new_liftoff, old_contact, state["last_contact_time"])
            new_air = torch.where(sample_contact, torch.zeros_like(old_air), old_air + dt)
            new_contact_time = torch.where(sample_contact, old_contact + dt, torch.zeros_like(old_contact))

            state["last_air_time"] = torch.where(active, new_last_air, state["last_air_time"])
            state["last_contact_time"] = torch.where(active, new_last_contact, state["last_contact_time"])
            state["current_air_time"] = torch.where(active, new_air, state["current_air_time"])
            state["current_contact_time"] = torch.where(active, new_contact_time, state["current_contact_time"])
            state["time_since_liftoff"] = torch.where(
                active,
                torch.where(new_liftoff, torch.zeros_like(state["time_since_liftoff"]), state["time_since_liftoff"] + dt),
                state["time_since_liftoff"],
            )
            state["time_since_touchdown"] = torch.where(
                active,
                torch.where(new_touchdown, torch.zeros_like(state["time_since_touchdown"]), state["time_since_touchdown"] + dt),
                state["time_since_touchdown"],
            )
            state["liftoff"] |= active & new_liftoff
            state["touchdown"] |= active & new_touchdown
            state["previous_contact"] = torch.where(active, sample_contact, state["previous_contact"])

    state["last_episode_step"] = torch.where(
        (advanced | init_env_mask), episode_step, state["last_episode_step"]
    )

    data_valid = state["initialized"] & valid_history[:, -1, :]
    result = {
        "contact": state["previous_contact"],
        "current_air_time": state["current_air_time"],
        "current_contact_time": state["current_contact_time"],
        "last_air_time": state["last_air_time"],
        "last_contact_time": state["last_contact_time"],
        "touchdown": state["touchdown"],
        "liftoff": state["liftoff"],
        "time_since_liftoff": state["time_since_liftoff"],
        "time_since_touchdown": state["time_since_touchdown"],
        "data_valid": data_valid,
        "initialized": state["initialized"],
    }
    env._robotdog_filtered_foot_timing_cache = {
        "step_key": step_key,
        "result": result,
    }
    return result


def orientation_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_angle: float = math.radians(5.0),
):
    """
    Penalizare continuă pentru înclinarea robotului.

    0 penalty = robot aproape vertical.
    Peste safe_angle, penalizarea crește progresiv.
    """

    robot = env.scene[asset_cfg.name]

    gravity_b = robot.data.projected_gravity_b

    # Unghiul total față de verticală.
    tilt_angle = torch.acos(
        torch.clamp(
            -gravity_b[:, 2],
            min=-1.0,
            max=1.0,
        )
    )

    excess_tilt = torch.relu(tilt_angle - safe_angle)

    return torch.square(excess_tilt / safe_angle)


def servo3_contact_penalty(
    env,
    force_scale: float = 10.0,
    contact_cost: float = 1.0,
    contact_threshold_n: float = GROUND_CONTACT_FORCE_THRESHOLD_N,
):
    """Penalize only servo3-to-terrain contacts; average over four legs."""
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive")
    if contact_threshold_n < 0.0:
        raise ValueError("contact_threshold_n must be non-negative")

    per_servo_costs = []
    per_servo_contacts = []
    for sensor_name in SERVO3_SENSOR_NAMES:
        force = _filtered_terrain_force_magnitude(env, sensor_name)
        in_contact = force > contact_threshold_n
        force_ratio = torch.clamp(force / force_scale, min=0.0, max=2.0)
        force_penalty = 0.25 * force_ratio.square() * in_contact.float()
        per_servo_contacts.append(in_contact)
        per_servo_costs.append(contact_cost * in_contact.float() + force_penalty)

    # Mean keeps the scale stable when one, two, three or four servo3 links touch.
    per_servo_costs = torch.stack(per_servo_costs, dim=-1)
    per_servo_contacts = torch.stack(per_servo_contacts, dim=-1)
    penalty = per_servo_costs.mean(dim=-1)
    return penalty


def foot_slip_penalty(
    env,
    velocity_scale: float = 0.20,
    servo3_slip_multiplier: float = 1.0,
    contact_threshold_n: float = GROUND_CONTACT_FORCE_THRESHOLD_N,
):
    """Penalize horizontal link-origin slip only while the link contacts terrain."""
    if velocity_scale <= 0.0:
        raise ValueError("velocity_scale must be positive")

    robot = env.scene["robot"]
    foot_body_ids, foot_body_names = robot.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    if list(foot_body_names) != list(FOOT_BODY_NAMES):
        raise RuntimeError(f"Foot body mapping mismatch: {foot_body_names}")
    servo3_body_ids, servo3_body_names = robot.find_bodies(SERVO3_BODY_NAMES, preserve_order=True)
    if list(servo3_body_names) != list(SERVO3_BODY_NAMES):
        raise RuntimeError(f"Servo3 body mapping mismatch: {servo3_body_names}")

    foot_costs = []
    servo3_costs = []
    for body_id, sensor_name in zip(foot_body_ids, FOOT_SENSOR_NAMES):
        in_contact = _filtered_terrain_contact_mask(env, sensor_name, contact_threshold_n)
        # body_link_lin_vel_w measures the same link-frame point as body_pos_w.
        velocity_xy = robot.data.body_link_lin_vel_w[:, int(body_id), :2]
        speed_xy = torch.linalg.vector_norm(velocity_xy, dim=-1)
        foot_costs.append(torch.square(speed_xy / velocity_scale) * in_contact.float())

    for body_id, sensor_name in zip(servo3_body_ids, SERVO3_SENSOR_NAMES):
        in_contact = _filtered_terrain_contact_mask(env, sensor_name, contact_threshold_n)
        velocity_xy = robot.data.body_link_lin_vel_w[:, int(body_id), :2]
        speed_xy = torch.linalg.vector_norm(velocity_xy, dim=-1)
        servo3_costs.append(torch.square(speed_xy / velocity_scale) * in_contact.float())

    foot_cost = torch.stack(foot_costs, dim=-1).mean(dim=-1)
    servo3_cost = torch.stack(servo3_costs, dim=-1).mean(dim=-1)
    total_slip_cost = foot_cost + servo3_slip_multiplier * servo3_cost

    curriculum_state = _initialize_terrain_curriculum_state(env)
    curriculum_state["slip_penalty_sum"] += (
        torch.clamp(total_slip_cost.detach(), min=0.0, max=5.0) * POLICY_DT
    )

    return total_slip_cost


def foot_impact_penalty(
    env,
    force_scale: float = 80.0,
    contact_threshold_n: float = GROUND_CONTACT_FORCE_THRESHOLD_N,
):
    """Penalize large normal impacts between componentfinal soles and terrain."""
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive")
    per_foot_cost = []
    for sensor_name in FOOT_SENSOR_NAMES:
        force = _filtered_terrain_force_magnitude(env, sensor_name)
        in_contact = force > contact_threshold_n
        safe_force = 0.25 * force_scale
        excess = torch.relu(force - safe_force) / force_scale
        per_foot_cost.append(excess.square() * in_contact.float())
    return torch.stack(per_foot_cost, dim=-1).mean(dim=-1)


def _robot(env):
    return env.scene["robot"]


def _sensor(env, name: str):
    return env.scene[name]


def final_target_action_rate_l2(env):
    """
    Penalize the movement of the final commanded joint target after
    command deadband + target velocity/acceleration limiting.

    The actuator runs at PHYSICS_HZ (200 Hz), while the policy/reward runs
    at POLICY_HZ (50 Hz). DelayedDCMotor accumulates squared target motion
    during the four physics substeps, and this function consumes that
    accumulated motion once per policy step.

    Multiplication by DECIMATION makes constant-velocity target movement
    across one policy interval comparable to a standard policy-rate
    action-rate penalty, while still catching substep-level back-and-forth
    motion that a simple 50 Hz endpoint difference would miss.
    """
    robot = env.scene["robot"]
    actuator = robot.actuators["all_joints"]

    accumulated_sq = actuator._target_rate_sq_sum
    initialized = actuator._target_rate_reward_initialized

    # Normalize with the same per-joint action scales used by the policy.
    action_term = env.action_manager.get_term("joint_position")
    scale = action_term._scale

    if not torch.is_tensor(scale):
        scale = torch.as_tensor(
            scale,
            device=env.device,
            dtype=accumulated_sq.dtype,
        )

    normalized_sq = accumulated_sq / torch.square(scale).clamp_min(1.0e-12)
    penalty = torch.mean(normalized_sq, dim=-1) * DECIMATION

    # Ignore the first policy interval after reset.
    penalty = torch.where(
        initialized,
        penalty,
        torch.zeros_like(penalty),
    )

    # The accumulator belongs to exactly one policy step.
    actuator._target_rate_sq_sum.zero_()
    actuator._target_rate_reward_initialized.fill_(True)

    return penalty


def joint_acc_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalizează accelerația articulară mare."""
    robot = env.scene[asset_cfg.name]
    acc = robot.data.joint_acc[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(acc / 50.0), dim=-1)


def joint_velocity_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalizare moale pentru viteze articulare foarte mari."""
    robot = env.scene[asset_cfg.name]
    vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(vel / 8.0), dim=-1)


def joint_limit_proximity(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin_fraction: float = 0.10,
):
    """Cost moale când articulația intră în ultimii ~10% ai domeniului."""
    robot = env.scene[asset_cfg.name]

    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    names = [robot.joint_names[i] for i in asset_cfg.joint_ids]

    total = torch.zeros(q.shape[0], device=q.device)

    for local_i, joint_name in enumerate(names):
        if joint_name not in JOINT_LIMITS_RAD:
            # Dacă USD are altă denumire, lăsăm USD-ul să gestioneze limita.
            continue

        lower, upper = JOINT_LIMITS_RAD[joint_name]
        span = max(upper - lower, 1.0e-6)
        margin = margin_fraction * span

        d_lower = q[:, local_i] - lower
        d_upper = upper - q[:, local_i]

        cost_lower = torch.exp(-torch.relu(d_lower) / margin) * (d_lower < margin)
        cost_upper = torch.exp(-torch.relu(d_upper) / margin) * (d_upper < margin)

        total += torch.maximum(cost_lower, cost_upper)

    return total / max(len(names), 1)





def terrain_gait_prior_scale(env, minimum_scale: float = 0.35):
    """Scale optional gait regularization down as terrain difficulty rises.

    This currently softens only the weak contact-coordination preference and
    terrain-sensitive base-height shaping; it does not set foot phases.
    """
    if not 0.0 <= minimum_scale <= 1.0:
        raise ValueError("minimum_scale must be in [0, 1]")
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_levels"):
        return torch.ones(env.num_envs, device=env.device)
    max_level = max(int(terrain.cfg.terrain_generator.num_rows) - 1, 1)
    level_fraction = torch.clamp(
        terrain.terrain_levels.to(dtype=torch.float32) / float(max_level),
        min=0.0,
        max=1.0,
    )
    return 1.0 - (1.0 - minimum_scale) * level_fraction


def base_roll_pitch_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalizare mică pentru roll/pitch; nu forțează postura perfect plată."""
    robot = env.scene[asset_cfg.name]
    gravity_b = robot.data.projected_gravity_b

    return gravity_b[:, 0].square() + gravity_b[:, 1].square()


def base_angular_velocity_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalize roll/pitch angular motion without suppressing commanded yaw."""
    robot = env.scene[asset_cfg.name]
    omega_roll_pitch = robot.data.root_ang_vel_b[:, :2]
    return torch.mean(torch.square(omega_roll_pitch / 4.0), dim=-1)


def uncommanded_yaw_rate_penalty(
    env,
    command_name: str = "base_velocity",
    command_threshold: float = 0.04,
    dead_zone: float = 0.03,
    normalization_speed: float = 0.25,
):
    """Penalize unintended base yaw while the commanded yaw rate is zero.

    The dead zone ignores small simulator noise. The normalized cost is
    clipped to [0, 1], and the term is inactive whenever yaw is commanded.
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    commanded_yaw = torch.abs(command[:, 2])
    actual_yaw_rate = torch.abs(robot.data.root_ang_vel_b[:, 2])
    no_yaw_command = commanded_yaw <= command_threshold

    excess_yaw_rate = torch.relu(actual_yaw_rate - dead_zone)
    normalized_cost = torch.clamp(
        excess_yaw_rate / normalization_speed,
        min=0.0,
        max=1.0,
    )
    # Do not punish a short evasive turn when the scan shows a barrier above
    # the step-up curriculum. For 6-8 cm climbable obstacles this mask is false,
    # so normal yaw stabilization remains fully active.
    obstacle_ahead = terrain_obstacle_ahead_mask(env)
    obstacle_relief = torch.where(
        obstacle_ahead,
        torch.full_like(normalized_cost, 0.25),
        torch.ones_like(normalized_cost),
    )
    return normalized_cost * no_yaw_command.float() * obstacle_relief


def soft_base_height_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_ground_scan"),
    target_height: float = 0.18,
    height_scale: float = 0.04,
    min_valid_rays: int = 5,
):
    """Penalize base height relative to the local ground under the body.

    A fixed env-origin reference would penalize the robot for correctly
    stepping onto raised terrain. A 3x3 downward RayCaster estimates the
    local support surface with a median, robust to one or two missing rays.
    """
    if height_scale <= 0.0 or min_valid_rays < 1:
        raise ValueError("height_scale and min_valid_rays must be positive")
    robot = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(hit_z)
    valid_count = valid.sum(dim=-1)
    safe_hits = torch.where(valid, hit_z, torch.full_like(hit_z, float("nan")))
    local_ground_z = torch.nanmedian(safe_hits, dim=-1).values
    fallback_ground_z = env.scene.env_origins[:, 2]
    local_ground_z = torch.where(
        valid_count >= min_valid_rays,
        local_ground_z,
        fallback_ground_z,
    )
    root_height_rel = robot.data.root_pos_w[:, 2] - local_ground_z
    error = (root_height_rel - target_height) / max(height_scale, 1.0e-6)
    penalty = torch.square(error)
    return penalty * terrain_gait_prior_scale(env, minimum_scale=0.40)


def body_contact_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 25.0,
):
    """Generic all-collider penalty; not registered for the feet in RewardsCfg.

    For foot/servo3 ground-specific penalties, use the terrain-filtered helpers
    below rather than this aggregate net-force helper.
    """
    sensor = env.scene[sensor_cfg.name]
    force = torch.linalg.norm(sensor.data.net_forces_w, dim=-1)

    excess = torch.relu(force - threshold) / threshold
    return torch.mean(excess.square(), dim=-1)




def feet_air_time_reward(
    env,
    command_name: str = "base_velocity",
    target_air_time: float = 0.16,
    target_clearance: float = 0.025,
    min_air_time: float = 0.05,
    moving_command_threshold: float = 0.02,
):
    """Touchdown lift reward, paid only during measured net forward locomotion."""
    if target_air_time <= 0.0 or target_clearance <= 0.0:
        raise ValueError("target_air_time and target_clearance must be positive")
    if min_air_time < 0.0 or moving_command_threshold < 0.0:
        raise ValueError("min_air_time and moving_command_threshold must be non-negative")
    motion_gate = robotdog_motion_progress_gate(
        env, command_name=command_name, command_threshold=moving_command_threshold
    )
    moving = motion_gate > 0.0
    timing = _robotdog_filtered_foot_contact_timing(env)
    clearance, clearance_valid = _robotdog_collision_clearance_state(env)
    n = env.num_envs
    episode_step = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    if not hasattr(env, "_robotdog_airtime_height_state"):
        env._robotdog_airtime_height_state = {
            "peak_clearance": torch.zeros((n, 4), device=env.device),
            "opposite_support_seen": torch.zeros((n, 4), dtype=torch.bool, device=env.device),
            "last_episode_step": episode_step.clone(),
        }
    state = env._robotdog_airtime_height_state
    episode_reset = episode_step < state["last_episode_step"]
    if torch.any(episode_reset):
        state["peak_clearance"][episode_reset] = 0.0
        state["opposite_support_seen"][episode_reset] = False
    state["last_episode_step"].copy_(episode_step)

    contact = timing["contact"]
    pair_a, pair_b = _robotdog_diagonal_foot_pairs(env)
    support_a = contact[:, list(pair_a)].any(dim=-1)
    support_b = contact[:, list(pair_b)].any(dim=-1)
    opposite_support = torch.zeros_like(contact)
    opposite_support[:, list(pair_a)] = support_b.unsqueeze(-1).expand(-1, len(pair_a))
    opposite_support[:, list(pair_b)] = support_a.unsqueeze(-1).expand(-1, len(pair_b))
    airborne_valid = (~contact) & timing["data_valid"] & clearance_valid & moving.unsqueeze(-1)
    state["opposite_support_seen"] = torch.where(
        timing["liftoff"], torch.zeros_like(state["opposite_support_seen"]),
        state["opposite_support_seen"],
    )
    state["opposite_support_seen"] |= airborne_valid & opposite_support
    current_peak = torch.maximum(state["peak_clearance"], torch.clamp(clearance, min=0.0))
    state["peak_clearance"] = torch.where(airborne_valid, current_peak, state["peak_clearance"])

    valid_touchdown = (
        timing["touchdown"] & timing["data_valid"] & clearance_valid
        & state["opposite_support_seen"] & (timing["last_air_time"] >= min_air_time)
        & moving.unsqueeze(-1)
    )
    peak = state["peak_clearance"].clone()
    clearance_score = torch.clamp(peak / target_clearance, 0.0, 1.0)
    airtime_score = torch.clamp(timing["last_air_time"] / target_air_time, 0.0, 1.0)
    reward = ((clearance_score * airtime_score) * valid_touchdown.float()).sum(dim=-1) / 4.0
    reward = reward * motion_gate

    validated_touchdown = timing["touchdown"] & timing["data_valid"]
    state["peak_clearance"] = torch.where(validated_touchdown, torch.zeros_like(state["peak_clearance"]), state["peak_clearance"])
    state["opposite_support_seen"] = torch.where(validated_touchdown, torch.zeros_like(state["opposite_support_seen"]), state["opposite_support_seen"])

    return reward



def diagonal_event_sync_reward(
    env,
    command_name: str = "base_velocity",
    event_window_s: float = DIAGONAL_EVENT_MATCH_WINDOW_S,
    sigma_s: float = DIAGONAL_EVENT_MATCH_SIGMA_S,
    velocity_threshold: float = 0.05,
):
    """Event synchronization shaping, gated by measured net forward locomotion."""
    if event_window_s <= 0.0 or sigma_s <= 0.0:
        raise ValueError("event_window_s and sigma_s must be positive")
    motion_gate = robotdog_motion_progress_gate(
        env, command_name=command_name, command_threshold=velocity_threshold
    )
    moving = motion_gate > 0.0
    timing = _robotdog_filtered_foot_contact_timing(env)
    pair_a, pair_b = _robotdog_diagonal_foot_pairs(env)
    event_scores, event_matches = [], []
    for event_key, age_key in (("liftoff", "time_since_liftoff"), ("touchdown", "time_since_touchdown")):
        event_now, event_age = timing[event_key], timing[age_key]
        for i, j in (pair_a, pair_b):
            age_i, age_j = event_age[:, i], event_age[:, j]
            finite = torch.isfinite(age_i) & torch.isfinite(age_j)
            safe_i = torch.where(finite, age_i, torch.zeros_like(age_i))
            safe_j = torch.where(finite, age_j, torch.zeros_like(age_j))
            gap = torch.abs(safe_i - safe_j)
            recent = finite & (age_i <= event_window_s) & (age_j <= event_window_s)
            triggered = event_now[:, i] | event_now[:, j]
            valid = recent & triggered & timing["data_valid"][:, i] & timing["data_valid"][:, j] & moving
            score = torch.exp(-0.5 * torch.square(gap / sigma_s))
            event_scores.append(torch.where(valid, score, torch.zeros_like(score)))
            event_matches.append(valid.float())
    scores = torch.stack(event_scores, dim=-1)
    matches = torch.stack(event_matches, dim=-1)
    reward = scores.sum(dim=-1) / matches.sum(dim=-1).clamp_min(1.0)
    reward = reward * motion_gate
    return reward



def diagonal_alternation_reward(
    env,
    command_name: str = "base_velocity",
    velocity_threshold: float = 0.05,
):
    """Weak opposite-diagonal support preference, only during real locomotion."""
    motion_gate = robotdog_motion_progress_gate(
        env, command_name=command_name, command_threshold=velocity_threshold
    )
    timing = _robotdog_filtered_foot_contact_timing(env)
    pair_a, pair_b = _robotdog_diagonal_foot_pairs(env)
    contact = timing["contact"].float()
    valid = timing["data_valid"].all(dim=-1) & (motion_gate > 0.0)
    support_a = contact[:, list(pair_a)].mean(dim=-1)
    support_b = contact[:, list(pair_b)].mean(dim=-1)
    air_a, air_b = 1.0 - support_a, 1.0 - support_b
    reward = torch.maximum(air_a * support_b, air_b * support_a)
    return reward * valid.float() * motion_gate




FOOT_PARENT_HIP_NAMES = (
    "servo2_1__body1",                # foot_1
    "servo2_mirror__1__body1",         # foot_2
    "servo2_mirror___1__1__body1",    # foot_3
    "servo2__1__1__body1",             # foot_4
)


def _unwrap_usd_relative_transform(result):
    """Handle USD Python bindings that return a matrix or (matrix, reset_flag)."""
    if callable(getattr(result, "Transform", None)):
        return result
    if isinstance(result, (tuple, list)):
        matrices = [item for item in result if callable(getattr(item, "Transform", None))]
        if len(matrices) == 1:
            return matrices[0]
    raise RuntimeError(
        "Unsupported USD ComputeRelativeTransform return value; expected a Gf matrix."
    )


def _reduce_collision_support_points(points: np.ndarray, max_points: int = FOOT_COLLISION_SUPPORT_DIRECTIONS) -> tuple[np.ndarray, str]:
    """Keep convex-support vertices needed for the minimum-distance query.

    ConvexHull gives the exact support vertices of a mesh point cloud. If Qhull
    is unavailable or the geometry is degenerate, deterministic Fibonacci-sphere
    support sampling keeps the point count bounded while retaining extrema from
    many directions. The fallback is explicitly reported in startup diagnostics.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        raise RuntimeError(f"Collision mesh contains only {len(points)} finite vertices")
    points = np.unique(np.round(points, decimals=8), axis=0)
    # Preserve the complete lowest collision band: it is the part of the foot
    # most relevant to ground clearance and must not disappear during reduction.
    local_z_min = float(points[:, 2].min())
    sole_band_indices = np.flatnonzero(points[:, 2] <= local_z_min + 0.002)

    method = "all_unique"
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        support_points = points[np.asarray(hull.vertices, dtype=np.int64)]
        method = "convex_hull"
    except Exception:
        # Continue with deterministic support-direction sampling below.
        support_points = points
        method = "support_direction_fallback"

    if len(support_points) > max_points:
        idx = np.arange(max_points, dtype=np.float64)
        z = 1.0 - 2.0 * (idx + 0.5) / float(max_points)
        radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        theta = idx * (math.pi * (3.0 - math.sqrt(5.0)))
        directions = np.stack((radius * np.cos(theta), radius * np.sin(theta), z), axis=-1)
        support_indices = {int(np.argmin(support_points @ direction)) for direction in directions}
        # Explicit axial supports protect the lowest/highest extrema.
        for axis in np.eye(3, dtype=np.float64):
            support_indices.add(int(np.argmin(support_points @ axis)))
            support_indices.add(int(np.argmax(support_points @ axis)))
        support_points = support_points[sorted(support_indices)]
        method += "+fibonacci_support"

    # Always include the actual underside band, even if it is not part of the
    # convex hull (e.g. a non-convex collider with a recessed lower surface).
    if len(sole_band_indices):
        support_points = np.concatenate((support_points, points[sole_band_indices]), axis=0)
        support_points = np.unique(np.round(support_points, decimals=8), axis=0)
        method += "+preserved_sole_band"
    return support_points, method


def _load_robotdog_collision_points_local(env):
    """Load each foot's actual USD collision geometry in its rigid-body frame."""
    if hasattr(env, "_robotdog_collision_points_local"):
        return env._robotdog_collision_points_local

    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
    except Exception as exc:
        raise RuntimeError(
            "USD collision clearance requires pxr (the USD Python API) inside Isaac Sim."
        ) from exc

    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise RuntimeError(f"Could not open RobotDog USD for collision geometry: {USD_PATH}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError(f"Invalid USD metersPerUnit={meters_per_unit}")

    # Resolve leaf body names exactly once; fail rather than silently using a
    # similarly named visual-only or duplicated prim.
    prims_by_name = {name: [] for name in FOOT_BODY_NAMES}
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        name = str(prim.GetName())
        if name in prims_by_name:
            prims_by_name[name].append(prim)

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points_by_foot = []
    geometry_info = []
    for body_name in FOOT_BODY_NAMES:
        matches = prims_by_name[body_name]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one USD body prim named {body_name!r}; found {len(matches)}: "
                f"{[str(p.GetPath()) for p in matches]}"
            )
        foot_prim = matches[0]
        collision_prims = [
            prim for prim in Usd.PrimRange(foot_prim)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not collision_prims:
            raise RuntimeError(
                f"No UsdPhysics.CollisionAPI prim exists under foot body {foot_prim.GetPath()}"
            )

        def has_collision_ancestor(prim) -> bool:
            current = prim
            while current and current.IsValid():
                if current.HasAPI(UsdPhysics.CollisionAPI):
                    return True
                if current == foot_prim:
                    break
                current = current.GetParent()
            return False

        point_chunks = []
        source_names = []
        for prim in Usd.PrimRange(foot_prim):
            if not prim.IsA(UsdGeom.Mesh) or not has_collision_ancestor(prim):
                continue
            raw_points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if raw_points is None or len(raw_points) == 0:
                continue
            relative = _unwrap_usd_relative_transform(
                xform_cache.ComputeRelativeTransform(prim, foot_prim)
            )
            transformed = np.empty((len(raw_points), 3), dtype=np.float64)
            for point_index, point in enumerate(raw_points):
                p = relative.Transform(
                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                )
                transformed[point_index] = (float(p[0]), float(p[1]), float(p[2]))
            if np.isfinite(transformed).all():
                point_chunks.append(transformed * meters_per_unit)
                source_names.append(str(prim.GetPath()))

        fallback_used = False
        if point_chunks:
            points_local = np.concatenate(point_chunks, axis=0)
        else:
            # Exact USD bounds are retained only as a guarded fallback for
            # analytic primitives (e.g. Cube/Capsule) without mesh vertices.
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
            )
            corners = []
            for collider in collision_prims:
                bbox = bbox_cache.ComputeRelativeBound(collider, foot_prim)
                bounds = bbox.ComputeAlignedRange()
                lo = np.asarray(bounds.GetMin(), dtype=np.float64)
                hi = np.asarray(bounds.GetMax(), dtype=np.float64)
                if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
                    continue
                for x in (lo[0], hi[0]):
                    for y in (lo[1], hi[1]):
                        for z in (lo[2], hi[2]):
                            corners.append([x, y, z])
            if not corners:
                raise RuntimeError(
                    f"Could not extract collision vertices/bounds for {foot_prim.GetPath()}"
                )
            points_local = np.asarray(corners, dtype=np.float64) * meters_per_unit
            fallback_used = True
            source_names = ["USD_collision_BBox_fallback"]

        reduced_points, reduction_method = _reduce_collision_support_points(points_local)
        if len(reduced_points) < 4:
            raise RuntimeError(f"Foot {body_name}: collision support point set is degenerate")
        points_by_foot.append(torch.as_tensor(reduced_points, dtype=torch.float32, device=env.device))
        geometry_info.append({
            "body": body_name,
            "source_vertices": int(len(points_local)),
            "support_points": int(len(reduced_points)),
            "reduction": reduction_method,
            "bbox_fallback": bool(fallback_used),
            "sources": source_names,
        })

    env._robotdog_collision_points_local = points_by_foot
    env._robotdog_collision_geometry_info = geometry_info
    print("[RobotDog collision clearance] USD collision geometry loaded:")
    for item in geometry_info:
        print(
            f"  {item['body']}: {item['source_vertices']} source vertices -> "
            f"{item['support_points']} support points; reduction={item['reduction']}; "
            f"bbox_fallback={item['bbox_fallback']}"
        )
        if item["bbox_fallback"]:
            print(
                f"  WARNING: {item['body']} uses USD bounding-box fallback; "
                "validate this primitive's collider shape before interpreting mm-level clearance."
            )
    return points_by_foot


def _robotdog_collision_clearance_state(env):
    """Measure collision-surface-to-terrain clearance using O(points) grid lookup.

    The foot RayCasters use a regular 2-D grid aligned to foot yaw. Instead of
    constructing an [N, support_points, ray_count] distance tensor, each
    collision point is transformed into that grid frame and its nearest grid
    cell is calculated analytically. The selected hit is then gathered and its
    actual XY distance is checked against the same tolerance as before.

    This preserves the USD collision support points, 2 cm ray spacing, nearest
    ray criterion, conservative all-points coverage gate, and minimum vertical
    collision-surface gap while avoiding hundreds of millions of pairwise
    point/ray comparisons per policy step.
    """
    step_key = _robotdog_current_step_key(env)
    cache = getattr(env, "_robotdog_collision_clearance_cache", None)
    if cache is not None and cache["step_key"] == step_key:
        return cache["clearance"], cache["valid"]

    robot = env.scene["robot"]
    foot_body_ids, resolved_names = robot.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    if list(resolved_names) != list(FOOT_BODY_NAMES):
        raise RuntimeError(f"Foot body mapping mismatch: {resolved_names}")
    local_points_by_foot = _load_robotdog_collision_points_local(env)

    grid_size_x, grid_size_y = FOOT_CLEARANCE_RAY_GRID_SIZE_M
    grid_resolution = float(FOOT_CLEARANCE_RAY_RESOLUTION_M)
    grid_nx = int(round(grid_size_x / grid_resolution)) + 1
    grid_ny = int(round(grid_size_y / grid_resolution)) + 1
    expected_ray_count = grid_nx * grid_ny
    half_x = 0.5 * float(grid_size_x)
    half_y = 0.5 * float(grid_size_y)
    max_neighbor_dist_sq = FOOT_CLEARANCE_MAX_NEAREST_RAY_DISTANCE_M ** 2

    clearance = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    valid = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)

    # Isaac Lab GridPatternCfg(ordering="xy") stores x as the inner loop and y
    # as the outer loop, hence flat_index = iy * grid_nx + ix.
    grid_x_local = torch.linspace(-half_x, half_x, grid_nx, device=env.device)
    grid_y_local = torch.linspace(-half_y, half_y, grid_ny, device=env.device)
    pattern_x = grid_x_local.repeat(grid_ny)
    pattern_y = grid_y_local.repeat_interleave(grid_nx)

    validated = getattr(env, "_robotdog_clearance_grid_validated", set())

    for foot_index, (body_id, local_points) in enumerate(zip(foot_body_ids, local_points_by_foot)):
        ray_sensor_name = f"foot_{foot_index + 1}_clearance_rays"
        ray_sensor = env.scene[ray_sensor_name]
        hits = ray_sensor.data.ray_hits_w
        if hits is None or hits.ndim != 3 or hits.shape[0] != env.num_envs or hits.shape[-1] != 3:
            raise RuntimeError(
                f"{ray_sensor_name} returned invalid ray_hits_w; expected [N,R,3], "
                f"got {None if hits is None else tuple(hits.shape)}"
            )
        if hits.shape[1] != expected_ray_count:
            raise RuntimeError(
                f"{ray_sensor_name} returned {hits.shape[1]} rays, but the configured "
                f"{grid_size_x}x{grid_size_y} grid at {grid_resolution} m requires "
                f"{expected_ray_count}. Refusing an incorrect grid-index lookup."
            )

        point = local_points.to(device=env.device, dtype=torch.float32)
        q = robot.data.body_link_quat_w[:, int(body_id), :]
        pos = robot.data.body_link_pos_w[:, int(body_id), :]
        qw, qx, qy, qz = q.unbind(dim=-1)

        # Full rigid-body rotation: collision vertices follow foot roll/pitch/yaw.
        r20 = 2.0 * (qx * qz - qw * qy)
        r21 = 2.0 * (qy * qz + qw * qx)
        r22 = 1.0 - 2.0 * (qx.square() + qy.square())
        point_z_w = (
            pos[:, 2:3]
            + r20.unsqueeze(-1) * point[:, 0].unsqueeze(0)
            + r21.unsqueeze(-1) * point[:, 1].unsqueeze(0)
            + r22.unsqueeze(-1) * point[:, 2].unsqueeze(0)
        )

        r00 = 1.0 - 2.0 * (qy.square() + qz.square())
        r01 = 2.0 * (qx * qy - qw * qz)
        r02 = 2.0 * (qx * qz + qw * qy)
        r10 = 2.0 * (qx * qy + qw * qz)
        r11 = 1.0 - 2.0 * (qx.square() + qz.square())
        r12 = 2.0 * (qy * qz - qw * qx)
        point_x_w = (
            pos[:, 0:1]
            + r00.unsqueeze(-1) * point[:, 0].unsqueeze(0)
            + r01.unsqueeze(-1) * point[:, 1].unsqueeze(0)
            + r02.unsqueeze(-1) * point[:, 2].unsqueeze(0)
        )
        point_y_w = (
            pos[:, 1:2]
            + r10.unsqueeze(-1) * point[:, 0].unsqueeze(0)
            + r11.unsqueeze(-1) * point[:, 1].unsqueeze(0)
            + r12.unsqueeze(-1) * point[:, 2].unsqueeze(0)
        )

        # RayCaster ray_alignment="yaw": the ray-grid frame ignores foot roll
        # and pitch. Its center is the body origin plus the yaw-rotated sensor offset.
        yaw_angle = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        sin_yaw = torch.sin(yaw_angle)
        cos_yaw = torch.cos(yaw_angle)
        center_local_x, center_local_y, _ = FOOT_SOLE_RAY_CENTER_LOCAL_M[foot_index]
        ray_center_x = (
            pos[:, 0]
            + cos_yaw * float(center_local_x)
            - sin_yaw * float(center_local_y)
        )
        ray_center_y = (
            pos[:, 1]
            + sin_yaw * float(center_local_x)
            + cos_yaw * float(center_local_y)
        )

        dx_world = point_x_w - ray_center_x.unsqueeze(-1)
        dy_world = point_y_w - ray_center_y.unsqueeze(-1)
        point_grid_x = cos_yaw.unsqueeze(-1) * dx_world + sin_yaw.unsqueeze(-1) * dy_world
        point_grid_y = -sin_yaw.unsqueeze(-1) * dx_world + cos_yaw.unsqueeze(-1) * dy_world

        ix_float = (point_grid_x + half_x) / grid_resolution
        iy_float = (point_grid_y + half_y) / grid_resolution
        # Match torch.min's first-index tie behavior: exact half-cell ties choose
        # the lower x/y grid index. Clamp to edge cells; the actual XY-distance
        # gate below decides whether an out-of-grid point is still within 20 mm.
        ix_base = torch.ceil(ix_float - 0.5).to(torch.long).clamp(0, grid_nx - 1)
        iy_base = torch.ceil(iy_float - 0.5).to(torch.long).clamp(0, grid_ny - 1)

        # Only a 3x3 neighborhood can contain a ray within the 20 mm tolerance
        # on a 20 mm square grid. Gather these nine candidates and select the
        # nearest VALID one. This also preserves behavior when a ray misses.
        candidate_indices = []
        for dy_index in (-1, 0, 1):
            iy_candidate = (iy_base + dy_index).clamp(0, grid_ny - 1)
            for dx_index in (-1, 0, 1):
                ix_candidate = (ix_base + dx_index).clamp(0, grid_nx - 1)
                candidate_indices.append(iy_candidate * grid_nx + ix_candidate)
        candidate_indices = torch.stack(candidate_indices, dim=-1)  # [N,P,9]
        n_points = int(point.shape[0])
        flat_candidate_indices = candidate_indices.reshape(env.num_envs, -1)
        candidate_hits = torch.gather(
            hits,
            dim=1,
            index=flat_candidate_indices.unsqueeze(-1).expand(-1, -1, 3),
        ).reshape(env.num_envs, n_points, 9, 3)
        candidate_valid = torch.isfinite(candidate_hits).all(dim=-1)
        candidate_dist_sq = (
            (point_x_w.unsqueeze(-1) - candidate_hits[..., 0]).square()
            + (point_y_w.unsqueeze(-1) - candidate_hits[..., 1]).square()
        )
        candidate_dist_sq = candidate_dist_sq.masked_fill(~candidate_valid, float("inf"))
        nearest_dist_sq, nearest_candidate = candidate_dist_sq.min(dim=-1)
        nearest_hits = torch.gather(
            candidate_hits,
            dim=2,
            index=nearest_candidate.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
        ).squeeze(2)
        point_valid = (
            torch.isfinite(nearest_dist_sq)
            & (nearest_dist_sq <= max_neighbor_dist_sq)
            & torch.isfinite(nearest_hits).all(dim=-1)
        )
        point_gap = point_z_w - nearest_hits[..., 2]
        point_gap = point_gap.masked_fill(~point_valid, float("inf"))
        min_gap = point_gap.amin(dim=-1)
        point_coverage = point_valid.float().mean(dim=-1)
        foot_valid = (point_coverage >= 1.0) & torch.isfinite(min_gap)
        clearance[:, foot_index] = torch.where(foot_valid, min_gap, torch.zeros_like(min_gap))
        valid[:, foot_index] = foot_valid

        # Runtime guard: confirm the assumed grid layout and yaw-only offset
        # against actual ray-hit XY coordinates once per foot. This prevents
        # silently applying the fast path to a differently ordered sensor.
        if foot_index not in validated:
            sample_env_ids = sorted(set((0, env.num_envs // 2, env.num_envs - 1)))
            max_xy_error = 0.0
            for env_id in sample_env_ids:
                qw0, qx0, qy0, qz0 = (q[env_id, k] for k in range(4))
                yaw0 = torch.atan2(
                    2.0 * (qw0 * qz0 + qx0 * qy0),
                    1.0 - 2.0 * (qy0.square() + qz0.square()),
                )
                sin_yaw0 = torch.sin(yaw0)
                cos_yaw0 = torch.cos(yaw0)
                expected_x = (
                    ray_center_x[env_id]
                    + cos_yaw0 * pattern_x
                    - sin_yaw0 * pattern_y
                )
                expected_y = (
                    ray_center_y[env_id]
                    + sin_yaw0 * pattern_x
                    + cos_yaw0 * pattern_y
                )
                actual_xy = hits[env_id, :, :2]
                finite_xy = torch.isfinite(actual_xy).all(dim=-1)
                if finite_xy.any():
                    error = torch.linalg.vector_norm(
                        actual_xy[finite_xy]
                        - torch.stack((expected_x, expected_y), dim=-1)[finite_xy],
                        dim=-1,
                    ).amax()
                    max_xy_error = max(max_xy_error, float(error.item()))
            if max_xy_error > 0.002:
                raise RuntimeError(
                    f"{ray_sensor_name} grid-index validation failed: max ray XY mismatch "
                    f"{max_xy_error:.6f} m. The sensor's actual offset/alignment/order does "
                    "not match the fast regular-grid lookup; do not use approximate clearance."
                )
            validated.add(foot_index)
            env._robotdog_clearance_grid_validated = validated

    env._robotdog_collision_clearance_cache = {
        "step_key": step_key,
        "clearance": clearance,
        "valid": valid,
    }
    return clearance, valid


def _robotdog_foot_clearance_state(
    env,
    command_name,
    moving_command_threshold,
    frequency_hz: float = GAIT_FREQUENCY_HZ,
    swing_ratio: float = GAIT_SWING_RATIO,
):
    """Compatibility adapter for dormant legacy helpers; active rewards are event-driven."""
    del frequency_hz, swing_ratio
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > moving_command_threshold
    timing = _robotdog_filtered_foot_contact_timing(env)
    clearance, geometry_valid = _robotdog_collision_clearance_state(env)
    actual_air_mask = (~timing["contact"]) & timing["data_valid"] & geometry_valid
    return clearance, actual_air_mask, moving


def _robotdog_diagonal_foot_pairs(env):
    """Infer diagonal foot pairs from the actual USD hip positions in base frame."""
    if not hasattr(env, "_robotdog_clearance_diagonal_pairs"):
        robot = env.scene["robot"]
        hip_body_ids, resolved_names = robot.find_bodies(
            list(FOOT_PARENT_HIP_NAMES), preserve_order=True
        )
        if list(resolved_names) != list(FOOT_PARENT_HIP_NAMES):
            raise RuntimeError(
                "Weak-diagonal clearance could not map hip bodies to foot sensors: "
                f"{resolved_names}"
            )

        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        hip_positions_b = torch.stack(
            [
                math_utils.quat_apply_inverse(
                    root_quat_w,
                    robot.data.body_pos_w[:, int(body_id), :] - root_pos_w,
                )
                for body_id in hip_body_ids
            ],
            dim=1,
        ).mean(dim=0)

        # RobotDog's forward direction is local -X; order each axle by local Y.
        x_order = torch.argsort(hip_positions_b[:, 0])
        front = sorted(x_order[:2].tolist(), key=lambda i: float(hip_positions_b[i, 1].item()))
        rear = sorted(x_order[2:].tolist(), key=lambda i: float(hip_positions_b[i, 1].item()))
        x_sorted = hip_positions_b[x_order, 0]
        front_gap = abs(float((hip_positions_b[front[1], 1] - hip_positions_b[front[0], 1]).item()))
        rear_gap = abs(float((hip_positions_b[rear[1], 1] - hip_positions_b[rear[0], 1]).item()))
        if float((x_sorted[2] - x_sorted[1]).item()) < 1.0e-4 or min(front_gap, rear_gap) < 1.0e-4:
            raise RuntimeError(
                "Weak-diagonal clearance could not distinguish the robot's front, rear, "
                "and sides. Verify the RobotDog USD axes."
            )

        # Same geometry-derived diagonals as gait_coordination_reward.
        env._robotdog_clearance_diagonal_pairs = (
            (front[0], rear[1]),
            (front[1], rear[0]),
        )
    return env._robotdog_clearance_diagonal_pairs


def _robotdog_gait_phase_per_foot(
    env,
    frequency_hz: float = GAIT_FREQUENCY_HZ,
    phase_offset: float = GAIT_PHASE_OFFSET,
    startup_delay_s: float = GAIT_STARTUP_DELAY_S,
):
    """Return normalized per-foot gait phase with a brief standing startup.

    Before ``startup_delay_s`` the phase is held constant in stance. Thereafter
    the two geometry-derived diagonal pairs run 180 degrees out of phase.
    """
    if frequency_hz <= 0.0:
        raise ValueError("gait frequency_hz must be positive")
    if startup_delay_s < 0.0:
        raise ValueError("startup_delay_s must be non-negative")

    elapsed_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    gait_elapsed_s = torch.clamp(elapsed_s - startup_delay_s, min=0.0)
    phase_a = torch.remainder(gait_elapsed_s * frequency_hz + phase_offset, 1.0)
    phase_b = torch.remainder(phase_a + 0.5, 1.0)

    pair_a, pair_b = _robotdog_diagonal_foot_pairs(env)
    phase_per_foot = torch.zeros((env.num_envs, 4), device=env.device)
    phase_per_foot[:, list(pair_a)] = phase_a.unsqueeze(-1)
    phase_per_foot[:, list(pair_b)] = phase_b.unsqueeze(-1)
    return phase_per_foot


def robotdog_gait_phase_observation(env):
    """Per-foot sin/cos phase features for the policy and teacher actor."""
    phase = _robotdog_gait_phase_per_foot(env)
    angle = 2.0 * math.pi * phase
    # Shape [num_envs, 8]: sin/cos for each of the four feet.
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).reshape(env.num_envs, 8)


def foot_trajectory_tracking_reward(
    env,
    command_name: str = "base_velocity",
    frequency_hz: float = GAIT_FREQUENCY_HZ,
    swing_ratio: float = GAIT_SWING_RATIO,
    stride_length: float = GAIT_STRIDE_LENGTH_M,
    clearance_height: float = GAIT_CLEARANCE_M,
    sigma_x: float = 0.045,
    moving_command_threshold: float = 0.05,
    x_tolerance_m: float = 0.025,
    y_tolerance_m: float = 0.025,
    z_tolerance_m: float = 0.012,
    minimum_progress_ratio: float = 0.25,
    minimum_progress_speed_mps: float = 0.05,
    stance_contact_phase_margin: float = 0.15,
    swing_air_phase_start: float = 0.20,
    swing_air_phase_end: float = 0.80,
    maximum_stance_slip_mps: float = 0.08,
    violation_weight: float = 0.25,
):
    """Strict straight-line trot trajectory score with anti-exploitation gates.

    The reference is longitudinal in base-frame X, keeps each foot in its
    reset lateral lane (Y), and uses the existing terrain-relative clearance
    baseline for Z. Positive trajectory reward is issued only when all four
    feet simultaneously satisfy their position, phase/contact and stance-slip
    conditions AND the base makes measurable progress along the command.

    Outside the accepted X/Y corridor, this term returns a bounded negative
    shaping cost, never a positive trajectory bonus. Missing contact/baseline
    data disables scoring rather than being interpreted as airborne/contact.
    This is intentionally a flat-ground, straight-line trot prior; it should be
    weakened as terrain difficulty or non-forward commands increase.
    """
    if not 0.0 < swing_ratio < 1.0:
        raise ValueError("swing_ratio must be between 0 and 1")
    if stride_length <= 0.0 or clearance_height <= 0.0:
        raise ValueError("stride_length and clearance_height must be positive")
    if min(sigma_x, x_tolerance_m, y_tolerance_m, z_tolerance_m) <= 0.0:
        raise ValueError("trajectory tolerances and sigma_x must be positive")
    if not 0.0 <= minimum_progress_ratio <= 1.0:
        raise ValueError("minimum_progress_ratio must be in [0, 1]")
    if minimum_progress_speed_mps < 0.0 or maximum_stance_slip_mps < 0.0:
        raise ValueError("progress and slip thresholds must be non-negative")
    if not 0.0 <= stance_contact_phase_margin < 0.5:
        raise ValueError("stance_contact_phase_margin must be in [0, 0.5)")
    if not 0.0 <= swing_air_phase_start < swing_air_phase_end <= 1.0:
        raise ValueError("swing air phase window must satisfy 0 <= start < end <= 1")
    if violation_weight < 0.0:
        raise ValueError("violation_weight must be non-negative")

    robot = env.scene["robot"]
    foot_body_ids, foot_body_names = robot.find_bodies(
        FOOT_BODY_NAMES,
        preserve_order=True,
    )
    if list(foot_body_names) != FOOT_BODY_NAMES:
        raise RuntimeError(f"Foot body mapping mismatch: {foot_body_names}")

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    foot_pos_b = torch.stack(
        [
            math_utils.quat_apply_inverse(
                root_quat_w,
                robot.data.body_pos_w[:, int(body_id), :] - root_pos_w,
            )
            for body_id in foot_body_ids
        ],
        dim=1,
    )

    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_x = command[:, 0]
    command_y = command[:, 1]
    command_yaw = command[:, 2] if command.shape[-1] > 2 else torch.zeros_like(command_x)
    command_speed = torch.linalg.norm(command_xy, dim=-1)

    # This trajectory is currently defined only for straight sagittal-plane
    # travel. Do not silently pay it for lateral or turning commands.
    straight_command = (
        (torch.abs(command_y) <= 1.0e-3)
        & (torch.abs(command_yaw) <= 1.0e-3)
    )
    moving_command = (
        (torch.abs(command_x) > moving_command_threshold)
        & straight_command
    )
    forward_axis = torch.sign(command_x).unsqueeze(-1)

    phase = _robotdog_gait_phase_per_foot(env, frequency_hz=frequency_hz)
    in_swing = phase < swing_ratio
    swing_u = torch.clamp(phase / swing_ratio, min=0.0, max=1.0)
    stance_u = torch.clamp(
        (phase - swing_ratio) / (1.0 - swing_ratio),
        min=0.0,
        max=1.0,
    )

    # At the maximum initial command (0.20 m/s), 1 Hz and 38% swing, this gives
    # 0.20 * 0.62 / 1.0 = 0.124 m of fore-aft foot travel per cycle.
    stride_length_per_env = (
        torch.abs(command_x) * (1.0 - swing_ratio) / frequency_hz
    ).clamp(min=0.07, max=stride_length * 1.20)

    stride = stride_length_per_env.unsqueeze(-1)
    swing_dx = -forward_axis * (0.5 - swing_u) * stride
    stance_dx = forward_axis * (0.5 - stance_u) * stride
    target_dx = torch.where(in_swing, swing_dx, stance_dx)
    target_lift = torch.where(
        in_swing,
        clearance_height * torch.sin(math.pi * swing_u).square(),
        torch.zeros_like(swing_u),
    )

    # Anchor the trajectory to each foot's measured reset pose. The target
    # starts at the current pose, then moves continuously around the cycle.
    episode_step = env.episode_length_buf.to(device=env.device, dtype=torch.long)
    if not hasattr(env, "_robotdog_gait_trajectory_state"):
        env._robotdog_gait_trajectory_state = {
            "center_x": torch.zeros((env.num_envs, 4), device=env.device),
            "center_y": torch.zeros((env.num_envs, 4), device=env.device),
            "initialized": torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
            "last_episode_step": episode_step.clone(),
        }
    gait_state = env._robotdog_gait_trajectory_state
    episode_reset = episode_step < gait_state["last_episode_step"]
    if torch.any(episode_reset):
        gait_state["initialized"][episode_reset] = False
    gait_state["last_episode_step"].copy_(episode_step)

    needs_init = ~gait_state["initialized"]
    if torch.any(needs_init):
        gait_state["center_x"][needs_init] = (
            foot_pos_b[:, :, 0][needs_init] - target_dx[needs_init]
        )
        gait_state["center_y"][needs_init] = foot_pos_b[:, :, 1][needs_init]
        gait_state["initialized"][needs_init] = True

    target_x = gait_state["center_x"] + target_dx
    target_y = gait_state["center_y"]
    x_error = foot_pos_b[:, :, 0] - target_x
    y_error = foot_pos_b[:, :, 1] - target_y

    # Reuse the same validated terrain-relative vertical baseline as the
    # existing clearance rewards. Z is a hard bonus condition, not a second
    # positive height reward (swing_height already owns vertical shaping).
    lift_per_foot, actual_air_mask, _ = _robotdog_foot_clearance_state(
        env,
        command_name=command_name,
        moving_command_threshold=moving_command_threshold,
        frequency_hz=frequency_hz,
        swing_ratio=swing_ratio,
    )
    clearance_state = env._robotdog_clearance_state
    baseline_valid = clearance_state["baseline_valid"]
    sensor_data_valid = clearance_state["current_sensor_data_valid"]
    data_valid_per_foot = baseline_valid & sensor_data_valid
    all_foot_data_valid = data_valid_per_foot.all(dim=-1)

    z_error = lift_per_foot - target_lift

    # Enforce the contact schedule only away from lift-off/touchdown boundaries:
    # stance core must be planted; middle swing must actually be airborne.
    stance_core = (
        (~in_swing)
        & (stance_u >= stance_contact_phase_margin)
        & (stance_u <= 1.0 - stance_contact_phase_margin)
    )
    swing_core = (
        in_swing
        & (swing_u >= swing_air_phase_start)
        & (swing_u <= swing_air_phase_end)
    )
    actual_contact = (~actual_air_mask) & data_valid_per_foot
    phase_contact_ok = (
        ((~stance_core) | actual_contact)
        & ((~swing_core) | actual_air_mask)
    )

    # A stance foot should remain nearly stationary in world coordinates.
    foot_velocity_xy_w = torch.stack(
        [robot.data.body_link_lin_vel_w[:, int(body_id), :2] for body_id in foot_body_ids],
        dim=1,
    )
    stance_slip_speed = torch.linalg.vector_norm(foot_velocity_xy_w, dim=-1)
    stance_slip_ok = (~stance_core) | (stance_slip_speed <= maximum_stance_slip_mps)

    position_ok = (
        (torch.abs(x_error) <= x_tolerance_m)
        & (torch.abs(y_error) <= y_tolerance_m)
        & (torch.abs(z_error) <= z_tolerance_m)
    )
    per_foot_compliant = position_ok & phase_contact_ok & stance_slip_ok
    all_four_feet_compliant = per_foot_compliant.all(dim=-1)

    # Hard anti-stationary gate: the body must move in the commanded direction
    # by at least 25% of the requested speed (and at least 0.05 m/s).
    command_dir = command_xy / command_speed.clamp_min(1.0e-6).unsqueeze(-1)
    actual_xy_yaw = velocity_xy_in_yaw_frame(env)
    actual_speed_along_command = torch.sum(actual_xy_yaw * command_dir, dim=-1)
    required_progress_speed = torch.maximum(
        torch.full_like(command_speed, minimum_progress_speed_mps),
        command_speed * minimum_progress_ratio,
    )
    progress_ok = actual_speed_along_command >= required_progress_speed

    valid_command = moving_command & (command_speed > 1.0e-6)
    positive_reward_gate = (
        all_foot_data_valid
        & valid_command
        & progress_ok
        & all_four_feet_compliant
    )

    # Reward horizontal trajectory quality only after ALL four feet pass the
    # complete X/Y/Z + contact + stance-slip gate. Z is a pass/fail condition
    # here; the existing foot-clearance/swing-height terms own its shaping.
    normalized_error_sq = (
        torch.square(x_error / sigma_x)
        + torch.square(y_error / y_tolerance_m)
    )
    per_foot_quality = torch.exp(-0.5 * normalized_error_sq)
    tracking_quality = per_foot_quality.mean(dim=-1)
    positive_bonus = tracking_quality * positive_reward_gate.float()

    # Keep a smooth learning signal for X/Y violations, but it is always a
    # cost. This avoids a zero-gradient cliff outside the hard bonus corridor.
    x_excess = (
        torch.relu(torch.abs(x_error) - x_tolerance_m) / x_tolerance_m
    ).clamp(max=1.0)
    y_excess = (
        torch.relu(torch.abs(y_error) - y_tolerance_m) / y_tolerance_m
    ).clamp(max=1.0)
    trajectory_violation_cost = (x_excess.square() + y_excess.square()).mean(dim=-1)

    data_and_command_gate = all_foot_data_valid & valid_command
    reward = torch.where(
        data_and_command_gate,
        positive_bonus - violation_weight * trajectory_violation_cost,
        torch.zeros_like(positive_bonus),
    )
    reward = reward * terrain_gait_prior_scale(env)


    return reward


def foot_clearance_reward(
    env,
    command_name: str = "base_velocity",
    target_height: float = GAIT_CLEARANCE_M,
    moving_command_threshold: float = 0.02,
):
    """Dense foot-lift shaping, paid only while the base achieves net progress."""
    if target_height <= 0.0:
        raise ValueError("target_height must be positive")
    motion_gate = robotdog_motion_progress_gate(
        env, command_name=command_name, command_threshold=moving_command_threshold
    )
    moving = motion_gate > 0.0
    timing = _robotdog_filtered_foot_contact_timing(env)
    clearance, geometry_valid = _robotdog_collision_clearance_state(env)
    valid = timing["data_valid"] & geometry_valid
    contact = timing["contact"]
    pair_a, pair_b = _robotdog_diagonal_foot_pairs(env)
    support_a = contact[:, list(pair_a)].any(dim=-1)
    support_b = contact[:, list(pair_b)].any(dim=-1)
    opposite_support = torch.zeros_like(contact)
    opposite_support[:, list(pair_a)] = support_b.unsqueeze(-1).expand(-1, len(pair_a))
    opposite_support[:, list(pair_b)] = support_a.unsqueeze(-1).expand(-1, len(pair_b))
    active = (~contact) & valid & opposite_support & moving.unsqueeze(-1)
    per_foot = torch.clamp(torch.clamp(clearance, min=0.0) / target_height, 0.0, 1.0)
    reward = (per_foot * active.float()).sum(dim=-1) / valid.float().sum(dim=-1).clamp_min(1.0)
    reward = reward * motion_gate * valid.any(dim=-1).float()

    curriculum_state = _initialize_terrain_curriculum_state(env)
    curriculum_state["clearance_reward_sum"] += reward.detach() * POLICY_DT
    curriculum_state["quality_elapsed"] += POLICY_DT
    return reward



def _robotdog_swing_reference(env, swing_ratio, target_height, frequency_hz):
    """Return CPG phase, expected swing mask, normalized swing time and Z target."""
    if not 0.0 < swing_ratio < 1.0:
        raise ValueError("swing_ratio must be between 0 and 1")
    if target_height <= 0.0:
        raise ValueError("target_height must be positive")

    phase = _robotdog_gait_phase_per_foot(env, frequency_hz=frequency_hz)
    expected_swing = phase < swing_ratio
    swing_u = torch.clamp(phase / swing_ratio, min=0.0, max=1.0)
    target_lift = target_height * torch.sin(math.pi * swing_u).square()
    target_lift = torch.where(expected_swing, target_lift, torch.zeros_like(target_lift))
    return phase, expected_swing, swing_u, target_lift


def swing_height_tracking_penalty(
    env,
    command_name: str = "base_velocity",
    frequency_hz: float = GAIT_FREQUENCY_HZ,
    swing_ratio: float = GAIT_SWING_RATIO,
    target_height: float = GAIT_CLEARANCE_M,
    moving_command_threshold: float = 0.05,
):
    """Penalize vertical deviation from the CPG height profile during expected swing.

    The mask is driven by expected CPG swing, not by actual airborne contact
    state. A foot that stays on the ground is penalized, and overshooting the
    target height is penalized as well.
    """
    lift_per_foot, actual_air_mask, moving = _robotdog_foot_clearance_state(
        env,
        command_name,
        moving_command_threshold,
        frequency_hz=frequency_hz,
        swing_ratio=swing_ratio,
    )
    _, expected_swing, _, target_lift = _robotdog_swing_reference(
        env, swing_ratio, target_height, frequency_hz
    )
    baseline_valid = env._robotdog_clearance_state["baseline_valid"]
    active = expected_swing & baseline_valid & moving.unsqueeze(-1)

    normalized_height_error = torch.clamp(
        (lift_per_foot - target_lift) / max(target_height, 1.0e-6),
        min=-1.0,
        max=1.0,
    )
    per_foot_cost = normalized_height_error.square()
    active_f = active.float()
    count = active_f.sum(dim=-1).clamp_min(1.0)
    penalty = (per_foot_cost * active_f).sum(dim=-1) / count
    penalty = penalty * moving.float()


    return penalty * terrain_gait_prior_scale(env)


def swing_contact_penalty(
    env,
    command_name: str = "base_velocity",
    frequency_hz: float = GAIT_FREQUENCY_HZ,
    swing_ratio: float = GAIT_SWING_RATIO,
    moving_command_threshold: float = 0.05,
    mid_swing_start: float = 0.20,
    mid_swing_end: float = 0.80,
    min_contact_force_n: float = 0.25,
):
    """Penalize measurable foot-ground contact in the middle of expected swing."""
    if not 0.0 <= mid_swing_start < mid_swing_end <= 1.0:
        raise ValueError("mid_swing_start/end must satisfy 0 <= start < end <= 1")
    if min_contact_force_n < 0.0:
        raise ValueError("min_contact_force_n must be non-negative")

    _, expected_swing, swing_u, _ = _robotdog_swing_reference(
        env, swing_ratio, GAIT_CLEARANCE_M, frequency_hz
    )
    _, _, moving = _robotdog_foot_clearance_state(
        env,
        command_name,
        moving_command_threshold,
        frequency_hz=frequency_hz,
        swing_ratio=swing_ratio,
    )
    baseline_valid = env._robotdog_clearance_state["baseline_valid"]
    mid_swing = (
        expected_swing
        & (swing_u >= mid_swing_start)
        & (swing_u <= mid_swing_end)
    )
    active = mid_swing & baseline_valid & moving.unsqueeze(-1)

    contact_cost = torch.zeros((env.num_envs, 4), device=env.device)
    for i, sensor_name in enumerate(FOOT_SENSOR_NAMES):
        force_norm = _filtered_terrain_force_magnitude(env, sensor_name)
        detected = force_norm > max(min_contact_force_n, GROUND_CONTACT_FORCE_THRESHOLD_N)
        contact_cost[:, i] = detected.float()

    active_f = active.float()
    count = active_f.sum(dim=-1).clamp_min(1.0)
    penalty = (contact_cost * active_f).sum(dim=-1) / count
    penalty = penalty * moving.float()


    return penalty * terrain_gait_prior_scale(env)


def stance_lift_penalty(
    env,
    command_name: str = "base_velocity",
    frequency_hz: float = GAIT_FREQUENCY_HZ,
    swing_ratio: float = GAIT_SWING_RATIO,
    target_height: float = GAIT_CLEARANCE_M,
    allowed_lift_m: float = 0.004,
    moving_command_threshold: float = 0.05,
):
    """Penalize lifting a foot above its small tolerance during expected stance."""
    if allowed_lift_m < 0.0:
        raise ValueError("allowed_lift_m must be non-negative")

    lift_per_foot, _, moving = _robotdog_foot_clearance_state(
        env,
        command_name,
        moving_command_threshold,
        frequency_hz=frequency_hz,
        swing_ratio=swing_ratio,
    )
    phase = _robotdog_gait_phase_per_foot(env, frequency_hz=frequency_hz)
    expected_stance = phase >= swing_ratio
    baseline_valid = env._robotdog_clearance_state["baseline_valid"]
    active = expected_stance & baseline_valid & moving.unsqueeze(-1)

    excess = torch.clamp(
        torch.relu(lift_per_foot - allowed_lift_m) / max(target_height, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    per_foot_cost = excess.square()
    active_f = active.float()
    count = active_f.sum(dim=-1).clamp_min(1.0)
    penalty = (per_foot_cost * active_f).sum(dim=-1) / count
    penalty = penalty * moving.float()


    return penalty * terrain_gait_prior_scale(env)


def weak_diagonal_foot_clearance_reward(
    env,
    command_name: str = "base_velocity",
    target_height: float = 0.04,
    moving_command_threshold: float = 0.01,
):
    """Add lift incentive only to whichever diagonal pair currently clears lower."""
    if target_height <= 0.0:
        raise ValueError("target_height must be positive")

    lift_per_foot, swing_mask, moving = _robotdog_foot_clearance_state(
        env, command_name, moving_command_threshold
    )
    pair_a, pair_b = _robotdog_diagonal_foot_pairs(env)

    def pair_swing_mean(pair):
        pair_swing = swing_mask[:, pair]
        count = pair_swing.float().sum(dim=-1)
        mean_lift = (lift_per_foot[:, pair] * pair_swing.float()).sum(dim=-1)
        return mean_lift / count.clamp_min(1.0), count > 0.0

    lift_a, active_a = pair_swing_mean(pair_a)
    lift_b, active_b = pair_swing_mean(pair_b)
    choose_a = active_a & (~active_b | (lift_a <= lift_b))
    choose_b = active_b & (~active_a | (lift_b < lift_a))

    selected_pair = torch.zeros_like(swing_mask)
    selected_pair[:, pair_a[0]] = choose_a
    selected_pair[:, pair_a[1]] = choose_a
    selected_pair[:, pair_b[0]] = choose_b
    selected_pair[:, pair_b[1]] = choose_b
    selected_swing = selected_pair & swing_mask

    per_foot_reward = torch.clamp(
        lift_per_foot / target_height, min=0.0, max=1.0
    )
    selected_count = selected_swing.float().sum(dim=-1)
    reward = (
        (per_foot_reward * selected_swing.float()).sum(dim=-1)
        / selected_count.clamp_min(1.0)
    )
    reward = reward * moving.float() * (selected_count > 0.0).float()


    return reward



def foot_contact_support_reward(env, command_name: str = "base_velocity"):
    """Support preference while locomoting; four-foot standing pays no bonus."""
    contacts = _robotdog_filtered_foot_contact_timing(env)["contact"]
    support_score = torch.clamp(contacts.float().sum(dim=-1) / 2.0, 0.0, 1.0)
    return support_score * robotdog_motion_progress_gate(env, command_name=command_name)



def velocity_xy_in_yaw_frame(env):
    """Viteza orizontala exprimata in cadrul orientat doar dupa yaw."""
    robot = env.scene["robot"]

    # Viteza reala in coordonate globale, numai pe planul XY.
    vel_w_xy = robot.data.root_lin_vel_w[:, :2]

    # Quaternion Isaac Lab: [w, x, y, z].
    q = robot.data.root_quat_w
    qw, qx, qy, qz = q.unbind(dim=-1)

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy.square() + qz.square()),
    )

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    forward = c * vel_w_xy[:, 0] + s * vel_w_xy[:, 1]
    lateral = -s * vel_w_xy[:, 0] + c * vel_w_xy[:, 1]

    return torch.stack((forward, lateral), dim=-1)


def forward_velocity_deficit_penalty(
    env,
    command_name: str = "base_velocity",
    command_threshold: float = 0.08,
    speed_tolerance: float = 0.03,
    normalization_speed: float = 0.25,
):
    """
    Penalizează viteza înainte insuficientă numai când
    robotului i se cere să meargă înainte.
    Returnează un cost pozitiv, aplicat cu weight negativ.
    """

    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    # This robot physically faces local -X. Convert both command and measured
    # local-X velocity into a positive physical-forward speed.
    commanded_speed = torch.clamp(
        -command[:, 0],
        min=0.0,
    )

    actual_speed = -velocity_xy_in_yaw_frame(env)[:, 0]

    active_command = commanded_speed > command_threshold

    # Permitem o eroare mică fără penalizare.
    speed_deficit = torch.relu(
        commanded_speed - actual_speed - speed_tolerance
    )

    normalized_deficit = torch.clamp(
        speed_deficit / normalization_speed,
        min=0.0,
        max=1.0,
    )

    obstacle_ahead = terrain_obstacle_ahead_mask(env)
    obstacle_relief = torch.where(
        obstacle_ahead,
        torch.full_like(normalized_deficit, 0.15),
        torch.ones_like(normalized_deficit),
    )
    return normalized_deficit * active_command.float() * obstacle_relief


def yaw_velocity_deficit_penalty(
    env,
    command_name: str = "base_velocity",
    command_threshold: float = 0.04,
    speed_tolerance: float = 0.02,
    normalization_speed: float = 0.15,
):
    """
    Penalizează lipsa rotației în direcția comandată.
    Nu penalizează comenzile yaw neglijabile.
    """

    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    commanded_yaw = command[:, 2]
    commanded_speed = torch.abs(commanded_yaw)
    actual_yaw = robot.data.root_ang_vel_b[:, 2]

    active_command = commanded_speed > command_threshold

    # Viteza reală semnată în direcția solicitată.
    aligned_yaw_speed = actual_yaw * torch.sign(commanded_yaw)

    yaw_deficit = torch.relu(
        commanded_speed - aligned_yaw_speed - speed_tolerance
    )

    normalized_deficit = torch.clamp(
        yaw_deficit / normalization_speed,
        min=0.0,
        max=1.0,
    )

    return normalized_deficit * active_command.float()


def velocity_tracking_forward(
    env,
    command_name: str = "base_velocity",
):
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    # This robot's physical forward direction is local -X.
    commanded_forward_speed = torch.clamp(
        -command[:, 0],
        min=0.0,
    )

    # Express actual velocity as positive when moving along physical forward (-X).
    actual_forward_speed = -robot.data.root_lin_vel_b[:, 0]

    # Reward-ul este activ numai cand se cere deplasare inainte.
    active_forward_command = commanded_forward_speed > 0.02

    # Reward liniar, mic, limitat la viteza comandata.
    # Miscarea inapoi nu primeste reward.
    rewarded_forward_speed = torch.minimum(
        torch.clamp(actual_forward_speed, min=0.0),
        commanded_forward_speed,
    )

    return (
        rewarded_forward_speed
        * active_forward_command.float()
    )



def velocity_tracking_lateral(
    env,
    command_name: str = "base_velocity",
):
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    error = command[:, 1] - robot.data.root_lin_vel_b[:, 1]
    return torch.exp(-2.0 * error.square())



def velocity_tracking_yaw(
    env,
    command_name: str = "base_velocity",
    command_threshold: float = 0.04,
    sigma: float = 0.12,
):
    """
    Track signed commanded yaw rate with a smooth reward around the target.
    """

    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    commanded_yaw = command[:, 2]
    commanded_speed = torch.abs(commanded_yaw)
    actual_yaw = robot.data.root_ang_vel_b[:, 2]

    active_command = commanded_speed > command_threshold

    if sigma <= 0.0:
        raise ValueError("Yaw-tracking sigma must be positive.")
    yaw_error = actual_yaw - commanded_yaw
    tracking = torch.exp(-torch.square(yaw_error / sigma))
    stationary_baseline = torch.exp(-torch.square(commanded_yaw / sigma))
    reward = (tracking - stationary_baseline) / (1.0 - stationary_baseline).clamp_min(1.0e-5)
    return torch.clamp(reward, min=-1.0, max=1.0) * active_command.float()




def stand_still_when_zero_command(
    env,
    command_name: str = "base_velocity",
):
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    # Comanda de deplasare liniara trebuie sa fie zero.
    linear_command_speed = torch.linalg.norm(
        command[:, :2],
        dim=-1,
    )

    # Comanda de rotatie trebuie sa fie zero.
    yaw_command_speed = torch.abs(command[:, 2])

    standing_command = (
        (linear_command_speed < 1.0e-3)
        & (yaw_command_speed < 1.0e-3)
    )

    # Miscarea reala a bazei: translatie + rotatie.
    movement = (
        torch.abs(robot.data.root_lin_vel_b[:, 0])
        + torch.abs(robot.data.root_lin_vel_b[:, 1])
        + 0.5 * torch.abs(robot.data.root_ang_vel_b[:, 2])
    )

    return movement * standing_command.float()



def base_height_stability(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalizare pentru oscilații mari ale bazei, nu doar eroare absolută."""
    robot = env.scene[asset_cfg.name]

    vertical_speed = torch.abs(robot.data.root_lin_vel_b[:, 2])
    return torch.square(vertical_speed / 1.5)


# =============================================================================
# TERRAIN
# =============================================================================
#
# Important:
# `terrain_generator=` primește DIRECT TerrainGeneratorCfg.
# Nu facem wrapper class și nu încercăm să accesăm `.generator`.
#
# TerrainGeneratorCfg cu curriculum=True sortează progresiv dificultatea.
# Familiile de teren sunt mixate pentru generalizare.
# =============================================================================

@configclass
class RobotDogDiscreteObstaclesTerrainCfg(SubTerrainBaseCfg):
    """Climbable 2-8 cm boxes plus staggered high barriers with bypass gaps."""

    obstacle_width_range: tuple[float, float] = (0.10, 0.20)
    obstacle_height_range: tuple[float, float] = (0.020, 0.080)
    num_obstacles: int = 8
    path_barrier_count: int = 2
    path_barrier_gap_width_range: tuple[float, float] = (0.50, 0.60)
    path_barrier_thickness_range: tuple[float, float] = (0.12, 0.16)
    path_barrier_height_range: tuple[float, float] = (0.14, 0.16)
    path_barrier_span_m: float = 2.40
    platform_width: float = 0.90
    edge_margin: float = 0.15
    obstacle_spacing: float = 0.025


def _robotdog_discrete_obstacles_trimesh(
    difficulty: float,
    cfg: RobotDogDiscreteObstaclesTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate climbable blocks and a progressive two-wall lateral bypass task.

    The robot resets facing world +X (yaw ~= pi) while its physical forward
    direction is local -X. Barriers are therefore placed at increasing world X.
    Each wall spans most of the central corridor but contains a 50-60 cm gap;
    successive gaps alternate sides to create a simple slalom/bypass problem.
    High barriers are 14-16 cm tall and are intended to trigger bypass rather than step-up behavior.
    """
    size_x, size_y = (float(cfg.size[0]), float(cfg.size[1]))
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError(f"Discrete-obstacle terrain size must be positive, got {cfg.size}.")

    d = float(np.clip(difficulty, 0.0, 1.0))
    width_min, width_max = map(float, cfg.obstacle_width_range)
    height_min, height_max = map(float, cfg.obstacle_height_range)
    active_height_max = float(np.interp(d, (0.0, 1.0), (height_min, height_max)))
    gap_min, gap_max = map(float, cfg.path_barrier_gap_width_range)
    depth_min, depth_max = map(float, cfg.path_barrier_thickness_range)
    barrier_height_min, barrier_height_max = map(float, cfg.path_barrier_height_range)
    margin = float(cfg.edge_margin)
    if size_x <= 2.0 * margin + width_min or size_y <= 2.0 * margin + width_min:
        raise ValueError("Discrete-obstacle terrain is too small for configured margins.")
    if not (0 <= int(cfg.path_barrier_count) <= 4):
        raise ValueError("path_barrier_count must be between 0 and 4.")
    if gap_min <= 0.0 or gap_max < gap_min:
        raise ValueError("Invalid path-barrier gap range.")
    if depth_min <= 0.0 or depth_max < depth_min:
        raise ValueError("Invalid path-barrier thickness range.")
    if barrier_height_min <= height_max or barrier_height_max < barrier_height_min:
        raise ValueError("Bypass barriers must be taller than the 2-8 cm climbable boxes.")

    meshes: list[trimesh.Trimesh] = []
    ground_thickness = 0.020
    ground = trimesh.creation.box(extents=(size_x, size_y, ground_thickness))
    ground.apply_translation((size_x * 0.5, size_y * 0.5, -0.5 * ground_thickness))
    meshes.append(ground)

    platform_width = min(float(cfg.platform_width), size_x - 2.0 * margin, size_y - 2.0 * margin)
    accepted_boxes: list[tuple[float, float, float, float]] = []
    reserved_gap_boxes: list[tuple[float, float, float, float]] = []

    def add_box(center_x: float, center_y: float, width_x: float, width_y: float, height: float) -> None:
        # Bottom is lifted 2 mm above the support plane to avoid coplanar faces.
        mesh_height = max(float(height) - 0.002, 0.005)
        mesh = trimesh.creation.box(extents=(width_x, width_y, mesh_height))
        mesh.apply_translation((center_x, center_y, 0.002 + 0.5 * mesh_height))
        meshes.append(mesh)
        accepted_boxes.append((center_x, center_y, width_x, width_y))

    def overlaps_platform(center_x: float, center_y: float, width_x: float, width_y: float) -> bool:
        return (
            abs(center_x - size_x * 0.5) < 0.5 * (platform_width + width_x)
            and abs(center_y - size_y * 0.5) < 0.5 * (platform_width + width_y)
        )

    def overlaps_regions(
        center_x: float, center_y: float, width_x: float, width_y: float
    ) -> bool:
        # Reserve the wall openings so random boxes cannot accidentally close
        # the only deliberately generated lateral bypass.
        for old_x, old_y, old_wx, old_wy in accepted_boxes + reserved_gap_boxes:
            separated_x = abs(center_x - old_x) >= 0.5 * (width_x + old_wx) + cfg.obstacle_spacing
            separated_y = abs(center_y - old_y) >= 0.5 * (width_y + old_wy) + cfg.obstacle_spacing
            if not (separated_x or separated_y):
                return True
        return False

    # Barriers appear only in the upper half of the curriculum. Their height
    # increases gradually from 14 to 16 cm; the 2-8 cm obstacles remain the
    # climb-over benchmark.
    barrier_progress = float(np.clip((d - 0.40) / 0.60, 0.0, 1.0))
    active_barriers = int(np.floor(barrier_progress * int(cfg.path_barrier_count) + 1.0e-6))
    if barrier_progress > 0.0 and active_barriers == 0 and int(cfg.path_barrier_count) > 0:
        active_barriers = 1
    barrier_height_cap = float(
        np.interp(barrier_progress, (0.0, 1.0), (barrier_height_min, barrier_height_max))
    )

    for barrier_index in range(active_barriers):
        barrier_depth = float(np.random.uniform(depth_min, depth_max))
        gap_width = float(np.random.uniform(gap_min, gap_max))
        barrier_x = size_x * 0.5 + 0.70 + 1.20 * barrier_index
        side = -1.0 if barrier_index % 2 == 0 else 1.0
        gap_center_y = size_y * 0.5 + side * 0.40
        span = min(float(cfg.path_barrier_span_m), size_y - 2.0 * margin)
        span_min_y = size_y * 0.5 - 0.5 * span
        span_max_y = size_y * 0.5 + 0.5 * span
        gap_start_y = gap_center_y - 0.5 * gap_width
        gap_end_y = gap_center_y + 0.5 * gap_width
        if not (span_min_y < gap_start_y < gap_end_y < span_max_y):
            raise ValueError("Path-barrier gap does not fit inside the configured wall span.")
        if barrier_x - 0.5 * barrier_depth < margin or barrier_x + 0.5 * barrier_depth > size_x - margin:
            raise ValueError("Path barrier does not fit inside the terrain tile along X.")

        barrier_height = float(np.random.uniform(barrier_height_min, max(barrier_height_min, barrier_height_cap)))
        for segment_start, segment_end in (
            (span_min_y, gap_start_y),
            (gap_end_y, span_max_y),
        ):
            segment_width_y = segment_end - segment_start
            if segment_width_y <= 0.02:
                continue
            segment_center_y = 0.5 * (segment_start + segment_end)
            add_box(barrier_x, segment_center_y, barrier_depth, segment_width_y, barrier_height)

        # A 60 cm-deep clear approach/exit corridor protects the gap from
        # random boxes while leaving the rest of the tile richly populated.
        reserved_gap_boxes.append(
            (barrier_x, gap_center_y, 0.90, gap_width + 0.18)
        )

    # Additional scattered climbable boxes. The spawn platform and bypass
    # corridors are kept clear to make failures informative rather than random.
    random_boxes_added = 0
    attempts = 0
    target_random_boxes = int(cfg.num_obstacles)
    max_attempts = max(500, target_random_boxes * 150)
    while random_boxes_added < target_random_boxes and attempts < max_attempts:
        attempts += 1
        width_x = float(np.random.uniform(width_min, width_max))
        width_y = float(np.random.uniform(width_min, width_max))
        center_x = float(np.random.uniform(margin + width_x * 0.5, size_x - margin - width_x * 0.5))
        center_y = float(np.random.uniform(margin + width_y * 0.5, size_y - margin - width_y * 0.5))
        if overlaps_platform(center_x, center_y, width_x, width_y):
            continue
        if overlaps_regions(center_x, center_y, width_x, width_y):
            continue
        height = float(np.random.uniform(height_min, max(active_height_max, height_min)))
        add_box(center_x, center_y, width_x, width_y, height)
        random_boxes_added += 1

    if random_boxes_added != target_random_boxes:
        raise RuntimeError(
            f"Could place only {random_boxes_added}/{target_random_boxes} random obstacles "
            f"after {attempts} attempts. Increase tile size or reduce spacing."
        )

    combined = trimesh.util.concatenate(meshes)
    terrain_origin = np.asarray((size_x * 0.5, size_y * 0.5, 0.0), dtype=np.float64)
    return [combined], terrain_origin


@configclass
class RobotDogSteppingStonesTerrainCfg(SubTerrainBaseCfg):
    """Low-polygon stepping-stone terrain made from simple box meshes."""

    stone_width_range: tuple[float, float] = (0.14, 0.24)
    gap_range: tuple[float, float] = (0.015, 0.060)
    stone_height_max: float = 0.025
    holes_depth: float = -0.040
    platform_width: float = 0.80
    edge_margin: float = 0.03


def _robotdog_stepping_stones_trimesh(
    difficulty: float,
    cfg: RobotDogSteppingStonesTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate real gaps/stepping surfaces with a compact box-based trimesh.

    A thin lower catch floor is retained at holes_depth, matching the previous
    stepping-stones task. Each stone is a simple cuboid; no dense height grid is
    generated. Stone width decreases, gaps widen, and vertical variation rises
    with curriculum difficulty.
    """
    size_x, size_y = (float(cfg.size[0]), float(cfg.size[1]))
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError(f"Stepping-stone terrain size must be positive, got {cfg.size}.")

    d = float(np.clip(difficulty, 0.0, 1.0))
    # Easy: broad stones and small gaps. Hard: narrower stones and wider gaps.
    configured_min, configured_max = map(float, cfg.stone_width_range)
    configured_gap_min, configured_gap_max = map(float, cfg.gap_range)
    stone_min = float(np.interp(
        d, (0.0, 1.0),
        (configured_max - 0.5 * (configured_max - configured_min), configured_min),
    ))
    stone_max = float(np.interp(
        d, (0.0, 1.0),
        (configured_max, configured_min + 0.5 * (configured_max - configured_min)),
    ))
    gap_min = float(np.interp(
        d, (0.0, 1.0),
        (configured_gap_min, configured_gap_min + 0.020),
    ))
    gap_max = float(np.interp(
        d, (0.0, 1.0),
        (configured_gap_min + (configured_gap_max - configured_gap_min) / 3.0, configured_gap_max),
    ))
    height_max = float(cfg.stone_height_max) * (0.25 + 0.75 * d)

    bottom_z = float(cfg.holes_depth)
    stone_base_z = bottom_z + 0.002  # tiny separation avoids coplanar overlapping faces
    margin = float(cfg.edge_margin)
    meshes: list[trimesh.Trimesh] = []

    # Lower surface at -4 cm: falling between stones is still a real drop,
    # but does not send the robot into an infinitely deep hole.
    floor_thickness = 0.010
    floor = trimesh.creation.box(
        extents=(size_x, size_y, floor_thickness)
    )
    floor.apply_translation(
        (size_x * 0.5, size_y * 0.5, bottom_z - floor_thickness * 0.5)
    )
    meshes.append(floor)

    # Central spawn/support platform, with its top at z=0.
    platform_w = min(float(cfg.platform_width), size_x - 2.0 * margin, size_y - 2.0 * margin)
    if platform_w <= 0.0:
        raise ValueError("Stepping-stone platform leaves no usable terrain area.")
    platform_h = max(-stone_base_z, 0.005)
    platform = trimesh.creation.box(
        extents=(platform_w, platform_w, platform_h)
    )
    platform.apply_translation(
        (size_x * 0.5, size_y * 0.5, stone_base_z + platform_h * 0.5)
    )
    meshes.append(platform)

    # Variable-size rows and columns produce irregular foot placements while
    # retaining a bounded 1.5-6 cm gap. The center platform remains unobstructed.
    y = margin
    row_index = 0
    while y + stone_min <= size_y - margin:
        stone_y = float(np.random.uniform(stone_min, stone_max))
        if y + stone_y > size_y - margin:
            break

        x = margin
        # Slightly stagger alternating rows without changing the local gap rule.
        if row_index % 2 == 1:
            x += float(np.random.uniform(0.015, min(0.060, gap_max)))

        while x + stone_min <= size_x - margin:
            stone_x = float(np.random.uniform(stone_min, stone_max))
            if x + stone_x > size_x - margin:
                break

            center_x = x + 0.5 * stone_x
            center_y = y + 0.5 * stone_y
            overlaps_spawn_platform = (
                abs(center_x - size_x * 0.5) < 0.5 * (platform_w + stone_x)
                and abs(center_y - size_y * 0.5) < 0.5 * (platform_w + stone_y)
            )

            if not overlaps_spawn_platform:
                top_z = float(np.random.uniform(0.0, max(height_max, 1.0e-4)))
                stone_height = max(top_z - stone_base_z, 0.005)
                stone = trimesh.creation.box(
                    extents=(stone_x, stone_y, stone_height)
                )
                stone.apply_translation(
                    (center_x, center_y, stone_base_z + 0.5 * stone_height)
                )
                meshes.append(stone)

            x += stone_x + float(np.random.uniform(gap_min, gap_max))

        y += stone_y + float(np.random.uniform(gap_min, gap_max))
        row_index += 1

    combined = trimesh.util.concatenate(meshes)
    terrain_origin = np.asarray(
        (size_x * 0.5, size_y * 0.5, 0.0), dtype=np.float64
    )
    return [combined], terrain_origin


# Terrain geometry strategy:
# - Trimesh: flat, repeated obstacles, stairs, and stepping stones.
# - Coarse Height Field (10 cm): only waves and slopes, where continuous
#   height variation is the actual learning signal.
# The original 6-family / 10-row / 18-column LP-ACRL task structure is retained.
TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(6.0, 6.0),
    border_width=20.0,
    num_rows=10,
    num_cols=18,
    difficulty_range=(0.0, 1.0),
    horizontal_scale=0.10,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.12,
        ),
        "waves": HfWaveTerrainCfg(
            proportion=0.18,
            amplitude_range=(0.005, 0.040),
            num_waves=5,
            horizontal_scale=0.10,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "discrete_obstacles": RobotDogDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            function=_robotdog_discrete_obstacles_trimesh,
            obstacle_width_range=(0.10, 0.20),
            obstacle_height_range=(0.020, 0.080),
            num_obstacles=8,
            path_barrier_count=2,
            path_barrier_gap_width_range=(0.50, 0.60),
            path_barrier_thickness_range=(0.12, 0.16),
            path_barrier_height_range=(0.14, 0.16),
            path_barrier_span_m=2.40,
            platform_width=0.90,
            edge_margin=0.15,
            obstacle_spacing=0.025,
        ),
        "stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.020, 0.080),
            step_width=0.20,
            platform_width=0.90,
            holes=False,
        ),
        "slopes": HfPyramidSlopedTerrainCfg(
            proportion=0.16,
            slope_range=(math.radians(5.0), math.radians(15.0)),
            platform_width=0.90,
            horizontal_scale=0.10,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "stepping_stones": RobotDogSteppingStonesTerrainCfg(
            proportion=0.18,
            function=_robotdog_stepping_stones_trimesh,
            stone_width_range=(0.14, 0.24),
            gap_range=(0.015, 0.060),
            stone_height_max=0.060,
            holes_depth=-0.040,
            platform_width=0.80,
            edge_margin=0.03,
        ),
    },
)

# =============================================================================
# SYNTHETIC TERRAIN-COMMAND CURRICULUM + LP-ACRL + ACS
# =============================================================================

# Forward speed is stored as a positive physical value. RobotDog's physical
# forward direction is local -X, so the actual command sent to the policy is
# negative on the command tensor's X component.
# Conservative initial envelope for 25 cm body / 20 cm maximum leg reach.
# LP-ACRL selects the feasible speed bin jointly with terrain family and row at reset.
# ACS ramps the command within that bin; it is not yet a per-obstacle reactive speed planner.
# 0.20 m/s = 0.8 body lengths/s = 1.0 leg-length/s; not a proven top speed.
SPEED_BIN_BOUNDS_MPS = (
    (0.04, 0.08),
    (0.08, 0.12),
    (0.12, 0.16),
    (0.16, 0.20),
)
NUM_SPEED_BINS = len(SPEED_BIN_BOUNDS_MPS)

# Direction modes: forward, reverse, left/right strafe, forward diagonals, pivot.
# The custom command term samples these modes per episode, then ramps each
# command smoothly. The physical forward direction is local -X.
MOTION_MODE_DIRECTIONS_XY = (
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (-0.70710678, 0.70710678),
    (-0.70710678, -0.70710678),
    (0.0, 0.0),
)
MOTION_MODE_PROBABILITIES = (0.42, 0.08, 0.12, 0.12, 0.08, 0.08, 0.10)
MOTION_MODE_PIVOT_INDEX = 6
YAW_COMMAND_ACTIVE_PROB = 0.35
YAW_COMMAND_MIN_RAD_S = 0.20
YAW_COMMAND_MAX_RAD_S = 0.60
PIVOT_YAW_MIN_RAD_S = 0.40
PIVOT_YAW_MAX_RAD_S = 0.80
ACS_YAW_SLEW_RAD_S_PER_STEP = 0.04  # 2 rad/s^2 at 50 Hz

# Approximate command moments from the motion-mode and LP-ACRL speed priors.
# The expanded warm-start uses affine compensation for X and initializes the
# previously constant Y/yaw dimensions with zero first-layer weights.
COMMAND_X_NORMALIZER_MEAN = -0.044
COMMAND_X_NORMALIZER_STD = 0.059
COMMAND_Y_NORMALIZER_MEAN = 0.0
COMMAND_Y_NORMALIZER_STD = 0.055
COMMAND_YAW_NORMALIZER_MEAN = 0.0
COMMAND_YAW_NORMALIZER_STD = 0.335

# Priors are normalized within each active axis/task family during setup.
LP_ACRL_LEVEL_PRIOR = (0.12, 0.14, 0.16, 0.16, 0.14, 0.10, 0.08, 0.05, 0.03, 0.02)
LP_ACRL_SPEED_PRIOR = (0.45, 0.30, 0.17, 0.08)
LP_ACRL_SCORE_EMA_ALPHA = 0.25
LP_ACRL_PROGRESS_EMA_ALPHA = 0.40
LP_ACRL_SOFTMAX_TEMPERATURE = 0.045
LP_ACRL_PRIOR_MIX = 0.20
LP_ACRL_MIN_EPISODES_PER_TASK = 4
LP_ACRL_LP_CLIP = 0.20

# Competence score; components are individually normalized to [0, 1].
LP_ACRL_SCORE_WEIGHT_TRACKING = 0.40
LP_ACRL_SCORE_WEIGHT_SURVIVAL = 0.25
LP_ACRL_SCORE_WEIGHT_CLEARANCE = 0.15
LP_ACRL_SCORE_WEIGHT_SLIP = 0.10
LP_ACRL_SCORE_WEIGHT_MOTOR = 0.10

# ACS ramp: 0.002 m/s per policy step at 50 Hz = 0.10 m/s^2.
# The ramp is measurement-independent; it never self-locks around a stationary robot.
ACS_COMMAND_SLEW_MPS_PER_STEP = 0.002

# Trainer-side PPO self-correction settings. The runner integration lives in
# robotdog_ppo_self_correction_v2.py because evaluation and optimizer control do
# not belong inside ManagerBasedRLEnvCfg.
PPO_SELF_CORRECTION_ENABLED = True
PPO_SELF_CORRECTION_EVAL_INTERVAL = 100
PPO_SELF_CORRECTION_PATIENCE = 2
PPO_SELF_CORRECTION_IMPROVEMENT_EPS = 0.005
PPO_SELF_CORRECTION_REGRESSION_TOL = 0.08
PPO_SELF_CORRECTION_LR_MULTIPLIER = 0.50
PPO_SELF_CORRECTION_MIN_LR = 1.0e-5
PPO_SELF_CORRECTION_STD_MULTIPLIER = 1.10
PPO_SELF_CORRECTION_MIN_STD = 0.05
PPO_SELF_CORRECTION_MAX_STD = 0.90

if abs(
    LP_ACRL_SCORE_WEIGHT_TRACKING
    + LP_ACRL_SCORE_WEIGHT_SURVIVAL
    + LP_ACRL_SCORE_WEIGHT_CLEARANCE
    + LP_ACRL_SCORE_WEIGHT_SLIP
    + LP_ACRL_SCORE_WEIGHT_MOTOR
    - 1.0
) > 1.0e-6:
    raise ValueError("LP-ACRL competence score weights must sum to 1.0")
if any(lo <= 0.0 or hi <= lo for lo, hi in SPEED_BIN_BOUNDS_MPS):
    raise ValueError("Every speed bin must have positive, strictly increasing bounds")
if abs(sum(MOTION_MODE_PROBABILITIES) - 1.0) > 1.0e-6:
    raise ValueError("MOTION_MODE_PROBABILITIES must sum to 1.0")
if len(MOTION_MODE_DIRECTIONS_XY) != len(MOTION_MODE_PROBABILITIES):
    raise ValueError("Every motion mode must have exactly one XY direction vector")
if SPEED_BIN_BOUNDS_MPS[-1][1] > 0.80 * ROBOT_BODY_LENGTH_M:
    raise ValueError("Maximum speed exceeds the initial 0.8 body-length/s envelope")
if GAIT_CLEARANCE_M > 0.25 * ROBOT_LEG_MAX_EXTENSION_M:
    raise ValueError("Swing clearance is too large relative to the available leg extension")
if ROBOT_LEG_MAX_EXTENSION_M <= 0.0:
    raise ValueError("Robot leg extension must be positive.")
if max(0.080, 0.080) >= ROBOT_LEG_MAX_EXTENSION_M * 0.50:
    raise ValueError("The configured 8 cm obstacle curriculum exceeds half the nominal maximum leg extension.")

# Local forward terrain preview for the terrain-aware teacher.
# The robot's physical forward direction is local -X.  The scan covers the
# region from roughly 5 cm to 85 cm ahead of the base and 1.60 m side-to-side.
# A 5 cm grid resolves the top surfaces of 6-8 cm step/box obstacles and both
# alternating lateral bypass gates much more reliably than the previous 10 cm grid.  An 18 cm height scale preserves 14-16 cm bypass-barrier relief without clipping.
TERRAIN_LOOKAHEAD_OFFSET_X_M = -0.45
TERRAIN_LOOKAHEAD_OFFSET_Z_M = 0.42
TERRAIN_LOOKAHEAD_SIZE_M = (0.80, 1.60)
TERRAIN_LOOKAHEAD_RESOLUTION_M = 0.05
TERRAIN_LOOKAHEAD_REFERENCE_BASE_HEIGHT_M = 0.18
TERRAIN_LOOKAHEAD_HEIGHT_SCALE_M = 0.18
TERRAIN_LOOKAHEAD_SAMPLES_X = int(round(TERRAIN_LOOKAHEAD_SIZE_M[0] / TERRAIN_LOOKAHEAD_RESOLUTION_M)) + 1
TERRAIN_LOOKAHEAD_SAMPLES_Y = int(round(TERRAIN_LOOKAHEAD_SIZE_M[1] / TERRAIN_LOOKAHEAD_RESOLUTION_M)) + 1
TERRAIN_LOOKAHEAD_RAY_COUNT = TERRAIN_LOOKAHEAD_SAMPLES_X * TERRAIN_LOOKAHEAD_SAMPLES_Y
TERRAIN_LOOKAHEAD_NOISE_STD_M = 0.003
TERRAIN_LOOKAHEAD_DROPOUT_PROB = 0.025
TERRAIN_OBSTACLE_DETOUR_THRESHOLD_M = 0.130

if TERRAIN_LOOKAHEAD_SIZE_M[1] < 1.50:
    raise ValueError("The lateral terrain scan must cover both staggered bypass gates.")
if TERRAIN_LOOKAHEAD_SAMPLES_X < 9 or TERRAIN_LOOKAHEAD_SAMPLES_Y < 17:
    raise ValueError("Terrain lookahead grid is too small for local obstacle and bypass detection.")
if TERRAIN_LOOKAHEAD_RAY_COUNT != 561:
    raise ValueError(f"Unexpected teacher terrain scan ray count: {TERRAIN_LOOKAHEAD_RAY_COUNT}; expected 561.")
if not (0.0 < TERRAIN_LOOKAHEAD_DROPOUT_PROB < 1.0):
    raise ValueError("Terrain lookahead dropout probability must be in (0, 1)")
if TERRAIN_LOOKAHEAD_NOISE_STD_M < 0.0:
    raise ValueError("Terrain lookahead noise standard deviation must be non-negative")


# =============================================================================
# SCENE
# =============================================================================




class DelayedDCMotor(DCMotor):
    """DCMotor with randomized command delay and backlash/hysteresis."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # ================================================================
        # DELAY BUFFERS
        # ================================================================
        self.positions_delay_buffer = DelayBuffer(
            cfg.max_delay,
            self._num_envs,
            device=self._device,
        )

        self.velocities_delay_buffer = DelayBuffer(
            cfg.max_delay,
            self._num_envs,
            device=self._device,
        )

        self.efforts_delay_buffer = DelayBuffer(
            cfg.max_delay,
            self._num_envs,
            device=self._device,
        )

        # ================================================================
        # BACKLASH STATE
        # ================================================================
        # Random backlash value for every environment and every joint.
        self.backlash = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        # Last delayed position command.
        self._last_command = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        # Effective target after backlash.
        self._backlash_target = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        # Previous movement direction:
        # -1 = negative
        #  0 = not initialized
        # +1 = positive
        self._direction = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        # Tracks whether the environment has received its first command.
        self._backlash_initialized = torch.zeros(
            self._num_envs,
            dtype=torch.bool,
            device=self._device,
        )


        # ================================================================
        # PER-JOINT ACTUATOR VARIATION
        # ================================================================

        # ================================================================
        # PER-JOINT ACTUATOR VARIATION
        # ================================================================

        # Independent torque variation for every environment and joint.
        self.torque_scale = torch.ones(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        # Independent speed variation for every environment and joint.
        self.speed_scale = torch.ones(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        # Mechanical zero/calibration offset for every servo.
        self.zero_offset = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )
        # ================================================================
        # SERVO COMMAND DEADBAND STATE
        # ================================================================
        # Last command actually accepted by the actuator/controller.
        self._last_servo_command = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        self._last_servo_command_initialized = torch.zeros(
            self._num_envs,
            dtype=torch.bool,
            device=self._device,
        )


        # ================================================================
        # TARGET VELOCITY / ACCELERATION LIMITER STATE
        # ================================================================
        # This tracks the smooth reference trajectory sent to the PD loop.

        self._rate_limited_target = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        self._rate_limited_target_velocity = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        self._rate_limited_target_initialized = torch.zeros(
            self._num_envs,
            dtype=torch.bool,
            device=self._device,
        )

        # ================================================================
        # FINAL TARGET RATE REWARD ACCUMULATOR
        # ================================================================
        # Accumulates squared motion of the FINAL reference target after
        # the command deadband + target velocity/acceleration limiter.
        #
        # compute() runs at 200 Hz, while the policy/reward runs at 50 Hz.
        # We therefore accumulate here and consume the result once per
        # policy step in final_target_action_rate_l2().
        self._target_rate_sq_sum = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )

        # Standard action-rate semantics ignore the first sample after reset.
        self._target_rate_reward_initialized = torch.zeros(
            self._num_envs,
            dtype=torch.bool,
            device=self._device,
        )


        # Debug: requested command delta and whether it was blocked.
        self._servo_command_requested_delta = torch.zeros(
            (self._num_envs, self.num_joints),
            device=self._device,
        )
        self._servo_command_blocked = torch.zeros(
            (self._num_envs, self.num_joints),
            dtype=torch.bool,
            device=self._device,
        )
        # ================================================================
        # FULL CONTROL-LOOP TRACE DEBUG
        # ================================================================
        # We keep the previous sample so we can reconstruct: 
        # target movement, actual movement, acceleration, target age,
        # movement age, torque direction changes and saturation.
        self._trace_counter = 0
        self._trace_max_prints = 240
        self._trace_print_every = 1  # Trace cadence is evaluated at 200 Hz physics rate.
        self._trace_target_threshold_rad = math.radians(0.05)
        self._trace_motion_velocity_threshold_rad_s = math.radians(0.5)

        self._trace_prev_q = torch.zeros(
            (self._num_envs, self.num_joints), device=self._device
        )
        self._trace_prev_qd = torch.zeros(
            (self._num_envs, self.num_joints), device=self._device
        )
        self._trace_prev_target = torch.zeros(
            (self._num_envs, self.num_joints), device=self._device
        )
        self._trace_prev_applied_tau = torch.zeros(
            (self._num_envs, self.num_joints), device=self._device
        )

        # Isaac Lab resolves actuator joints in articulation/USD order, which
        # need not match numeric revolute_1..revolute_12 order. Validate the
        # joint set, then keep a named permutation so telemetry is stored in
        # canonical numeric order. This preserves the downstream four
        # consecutive (hip, femur, tibia) leg triplets without changing the
        # actuator's native tensor ordering or joint commands.
        expected_joint_names = [f"revolute_{i}" for i in range(1, 13)]
        resolved_joint_names = list(self.joint_names)
        if (
            len(resolved_joint_names) != len(expected_joint_names)
            or len(set(resolved_joint_names)) != len(expected_joint_names)
            or set(resolved_joint_names) != set(expected_joint_names)
        ):
            raise RuntimeError(
                "Motor telemetry requires exactly revolute_1..revolute_12; "
                f"resolved {resolved_joint_names}. Check the USD articulation joint set."
            )
        telemetry_order = [resolved_joint_names.index(name) for name in expected_joint_names]
        self._robotdog_telemetry_joint_indices = torch.tensor(
            telemetry_order, dtype=torch.long, device=self._device
        )
        print(
            "[RobotDog motor telemetry] actuator joint order="
            f"{resolved_joint_names}; canonical telemetry order={expected_joint_names}; "
            f"permutation={telemetry_order}",
            flush=True,
        )

        # Per-environment, per-joint motor telemetry. A robot-wide average can
        # hide one overloaded servo, so values are stored in canonical numeric
        # revolute_1..revolute_12 order for correct per-leg aggregation.
        joint_shape = (self._num_envs, self.num_joints)
        self._robotdog_motor_telemetry = {
            "torque_util_sum": torch.zeros(joint_shape, device=self._device),
            "speed_util_sum": torch.zeros(joint_shape, device=self._device),
            "clip_deficit_sum": torch.zeros(joint_shape, device=self._device),
            "saturation_sum": torch.zeros(joint_shape, device=self._device),
            "episode_samples": torch.zeros(self._num_envs, device=self._device),
            "policy_torque_util_sum": torch.zeros(joint_shape, device=self._device),
            "policy_clip_deficit_sum": torch.zeros(joint_shape, device=self._device),
            "policy_saturation_sum": torch.zeros(joint_shape, device=self._device),
            "policy_samples": torch.zeros(self._num_envs, device=self._device),
        }

        self._trace_target_age_steps = torch.zeros(
            (self._num_envs, self.num_joints), dtype=torch.long, device=self._device
        )
        self._trace_motion_age_steps = torch.zeros(
            (self._num_envs, self.num_joints), dtype=torch.long, device=self._device
        )
        self._trace_initialized = torch.zeros(
            self._num_envs, dtype=torch.bool, device=self._device
        )

        # ================================================================
        # SERVO DEADBand DEBUG
        # ================================================================
        self._servo_debug_counter = 0
        self._servo_debug_print_every = 1
        self._servo_debug_max_prints = 0

        try:
            self._debug_joint_index = self.joint_names.index("revolute_10")
        except ValueError:
            self._debug_joint_index = 0

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)

        # ================================================================
        # RANDOMIZE DELAY
        # ================================================================
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)

        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        
               # ================================================================
        # RANDOMIZE ACTUATOR STRENGTH
        # ================================================================

        torque_scales = torch.empty(
            (num_envs, self.num_joints),
            device=self._device,
        ).uniform_(
            self.cfg.torque_scale_min,
            self.cfg.torque_scale_max,
        )

        # ================================================================
        # RANDOMIZE ACTUATOR SPEED
        # ================================================================

        speed_scales = torch.empty(
            (num_envs, self.num_joints),
            device=self._device,
        ).uniform_(
            self.cfg.speed_scale_min,
            self.cfg.speed_scale_max,
        )

        # ================================================================
        # RANDOMIZE JOINT ZERO / SERVO CALIBRATION
        # ================================================================

        zero_offsets = torch.empty(
            (num_envs, self.num_joints),
            device=self._device,
        ).uniform_(
            self.cfg.zero_offset_min_rad,
            self.cfg.zero_offset_max_rad,
        )

        if env_ids is None or env_ids == slice(None):
            self.torque_scale[:] = torque_scales
            self.speed_scale[:] = speed_scales
            self.zero_offset[:] = zero_offsets

            active_env_ids = slice(None)
        else:
            self.torque_scale[env_ids] = torque_scales
            self.speed_scale[env_ids] = speed_scales
            self.zero_offset[env_ids] = zero_offsets

            active_env_ids = env_ids



        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)

        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

        # ================================================================
        # RANDOMIZE BACKLASH
        # ================================================================
        backlash_values = torch.empty(
            (num_envs, self.num_joints),
            device=self._device,
        ).uniform_(
            self.cfg.backlash_min,
            self.cfg.backlash_max,
        )

        if env_ids is None or env_ids == slice(None):
            self.backlash[:] = backlash_values
            self._last_command.zero_()
            self._backlash_target.zero_()
            self._direction.zero_()
            self._backlash_initialized.zero_()
            self._last_servo_command.zero_()
            self._last_servo_command_initialized.zero_()
            self._servo_command_requested_delta.zero_()
            self._servo_command_blocked.zero_()
            self._rate_limited_target.zero_()
            self._rate_limited_target_velocity.zero_()
            self._rate_limited_target_initialized.zero_()
            self._target_rate_sq_sum.zero_()
            self._target_rate_reward_initialized.zero_()
            for telemetry_buffer in self._robotdog_motor_telemetry.values():
                telemetry_buffer.zero_()
        else:
            self.backlash[env_ids] = backlash_values
            self._last_command[env_ids] = 0.0
            self._backlash_target[env_ids] = 0.0
            self._direction[env_ids] = 0.0
            self._backlash_initialized[env_ids] = False
            self._last_servo_command[env_ids] = 0.0
            self._last_servo_command_initialized[env_ids] = False
            self._servo_command_requested_delta[env_ids] = 0.0
            self._servo_command_blocked[env_ids] = False
            self._rate_limited_target[env_ids] = 0.0
            self._rate_limited_target_velocity[env_ids] = 0.0
            self._rate_limited_target_initialized[env_ids] = False
            self._target_rate_sq_sum[env_ids] = 0.0
            self._target_rate_reward_initialized[env_ids] = False
            for telemetry_buffer in self._robotdog_motor_telemetry.values():
                telemetry_buffer[env_ids] = 0.0




    def _apply_backlash(self, command: torch.Tensor) -> torch.Tensor:
        """Apply a simple direction-dependent backlash model."""

        # ================================================================
        # INITIAL COMMAND
        # ================================================================
        new_envs = ~self._backlash_initialized

        if torch.any(new_envs):
            self._last_command[new_envs] = command[new_envs]
            self._backlash_target[new_envs] = command[new_envs]
            self._direction[new_envs] = 0.0
            self._backlash_initialized[new_envs] = True

        # Change in commanded position.
        delta = command - self._last_command

        moving = torch.abs(delta) > 1.0e-7
        new_direction = torch.sign(delta)

        # ================================================================
        # SAME DIRECTION
        # ================================================================
        same_direction = (
            moving
            & (self._direction != 0.0)
            & (new_direction == self._direction)
        )

        if torch.any(same_direction):
            self._backlash_target[same_direction] = command[same_direction]

        # ================================================================
        # DIRECTION REVERSAL
        # ================================================================
        reversing = (
            moving
            & (self._direction != 0.0)
            & (new_direction != self._direction)
        )

        if torch.any(reversing):
            distance_from_target = torch.abs(
                command - self._backlash_target
            )

            backlash_taken_up = (
                reversing
                & (distance_from_target >= self.backlash)
            )

            if torch.any(backlash_taken_up):
                self._direction[backlash_taken_up] = (
                    new_direction[backlash_taken_up]
                )

                sign = new_direction[backlash_taken_up]

                self._backlash_target[backlash_taken_up] = (
                    command[backlash_taken_up]
                    - sign * self.backlash[backlash_taken_up]
                )

        # ================================================================
        # FIRST MOVEMENT DIRECTION
        # ================================================================
        no_previous_direction = (
            moving
            & (self._direction == 0.0)
        )

        if torch.any(no_previous_direction):
            self._direction[no_previous_direction] = (
                new_direction[no_previous_direction]
            )

            self._backlash_target[no_previous_direction] = (
                command[no_previous_direction]
            )

        self._last_command = command

        return self._backlash_target


    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """
        DCMotor torque-speed clipping with per-environment/per-joint
        actuator strength and speed variation.

        The nominal Isaac Lab DCMotor model is preserved, but the motor
        characteristics are scaled independently for every servo.
        """

        # ---------------------------------------------------------------
        # Randomized motor characteristics
        # ---------------------------------------------------------------

        effective_saturation_effort = (
            self._saturation_effort
            * self.torque_scale
        )

        effective_effort_limit = (
            self.effort_limit
            * self.torque_scale
        )

        effective_velocity_limit = (
            self.velocity_limit
            * self.speed_scale
        )

        # Avoid division by zero.
        effective_velocity_limit = torch.clamp(
            effective_velocity_limit,
            min=1.0e-6,
        )

        # ---------------------------------------------------------------
        # Official Isaac Lab DCMotor torque-speed envelope
        # ---------------------------------------------------------------

        max_effort = (
            effective_saturation_effort
            * (
                1.0
                - self._joint_vel
                / effective_velocity_limit
            )
        )

        max_effort = torch.clip(
            max_effort,
            min=self._zeros_effort,
            max=effective_effort_limit,
        )

        min_effort = (
            effective_saturation_effort
            * (
                -1.0
                - self._joint_vel
                / effective_velocity_limit
            )
        )

        min_effort = torch.clip(
            min_effort,
            min=-effective_effort_limit,
            max=self._zeros_effort,
        )

        return torch.clip(
            effort,
            min=min_effort,
            max=max_effort,
        )

    # def compute(
    #     self,
    #     control_action: ArticulationActions,
    #     joint_pos: torch.Tensor,
    #     joint_vel: torch.Tensor,
    # ) -> ArticulationActions:

    #     # ================================================================
    #     # 1. COMMAND DELAY
    #     # ================================================================
    #     if control_action.joint_positions is not None:

    #         # Per-servo mechanical/calibration zero error.
    #         calibrated_position = (
    #             control_action.joint_positions + self.zero_offset
    #         )

    #         delayed_position = self.positions_delay_buffer.compute(
    #             calibrated_position
    #         )

    #         # ============================================================
    #         # 2. BACKLASH / HYSTERESIS
    #         # ============================================================
    #         control_action.joint_positions = self._apply_backlash(
    #             delayed_position
    #         )

    #     # These are optional for position control.
    #     if control_action.joint_velocities is not None:
    #         control_action.joint_velocities = (
    #             self.velocities_delay_buffer.compute(
    #                 control_action.joint_velocities
    #             )
    #         )

    #     if control_action.joint_efforts is not None:
    #         control_action.joint_efforts = (
    #             self.efforts_delay_buffer.compute(
    #                 control_action.joint_efforts
    #             )
    #         )

    #     # ================================================================
    #     # 3. NORMAL DC MOTOR MODEL
    #     # ================================================================
    #     return super().compute(
    #         control_action,
    #         joint_pos,
    #         joint_vel,
         #)







    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:

        # ================================================================
        # 1. COMMAND DELAY + CALIBRATION
        # ================================================================
        if control_action.joint_positions is not None:

            # Target produced by JointPositionAction after action scaling.
            raw_position = control_action.joint_positions.clone()

            # Servo zero/calibration offset.
            calibrated_position = (
                control_action.joint_positions + self.zero_offset
            )

            # Command delay.
            delayed_position = self.positions_delay_buffer.compute(
                calibrated_position
            )

            # ============================================================
            # 2. SERVO COMMAND DEADBand: 1.5 DEG
            # ============================================================
            # This operates INSIDE the actuator model.
            #
            # We compare the requested command with the LAST COMMAND
            # ACCEPTED by the servo/controller, not with the previous
            # physics sample and not with the physical joint position.
            #
            # < 1.5 deg -> reject new command, keep last accepted command.
            # >= 1.5 deg -> accept new command and update the reference.

            # ============================================================
            # SERVO COMMAND DEADBAND: 1.5 DEG
            # ============================================================

            # Snapshot al ultimei comenzi acceptate ÎNAINTE de decizie.
            previous_servo_command = self._last_servo_command.clone()

            # Prima comandă după reset este acceptată automat.
            new_envs = ~self._last_servo_command_initialized

            # Diferența dintre comanda candidată și ultima comandă acceptată.
            command_delta = delayed_position - previous_servo_command

            # Acceptăm dacă:
            #   - este prima comandă, SAU
            #   - schimbarea este >= 1.5°
            command_accepted = (
                new_envs.unsqueeze(-1)
                | (torch.abs(command_delta) >= SERVO_COMMAND_DEADBAND_RAD)
            )
            # Update numai pentru comenzile acceptate.
            self._last_servo_command = torch.where(
                command_accepted,
                delayed_position,
                previous_servo_command,
            )

            # Inițializare după prima comandă.
            self._last_servo_command_initialized = (
                self._last_servo_command_initialized
                | new_envs
            )

            # Debug state.
            self._servo_command_requested_delta = command_delta.clone()
            self._servo_command_blocked = ~command_accepted


            # Aceasta este comanda acceptată de deadband.
            requested_target = self._last_servo_command

            # ============================================================
            # TARGET VELOCITY + ACCELERATION LIMITER
            # ============================================================
            # This limits how quickly the commanded reference itself may move.
            #
            # IMPORTANT:
            # This is NOT a hard limit on the physical joint velocity.
            # It is a limit on the target trajectory sent to the PD controller.

            first_target = ~self._rate_limited_target_initialized

            current_target = torch.where(
                first_target.unsqueeze(-1),
                joint_pos,
                self._rate_limited_target,
            )

            previous_target_velocity = self._rate_limited_target_velocity

            target_error = requested_target - current_target

            # Velocity that would reach the target in one timestep,
            # limited to the configured maximum target velocity.
            desired_target_velocity = torch.clamp(
                target_error / PHYSICS_DT,
                min=-self.cfg.target_velocity_limit,
                max=self.cfg.target_velocity_limit,
            )

            # Limit how quickly the reference velocity itself can change.
            max_velocity_change = (
                self.cfg.target_acceleration_limit * PHYSICS_DT
            )

            velocity_change = torch.clamp(
                desired_target_velocity - previous_target_velocity,
                min=-max_velocity_change,
                max=max_velocity_change,
            )

            new_target_velocity = (
                previous_target_velocity + velocity_change
            )

            target_step = new_target_velocity * PHYSICS_DT

            # Never step past the requested target.
            reached_target = (
                torch.abs(target_step)
                >= torch.abs(target_error)
            )

            target_step = torch.where(
                reached_target,
                target_error,
                target_step,
            )

            new_target_velocity = torch.where(
                reached_target,
                torch.zeros_like(new_target_velocity),
                new_target_velocity,
            )

            new_target = current_target + target_step

            # ============================================================
            # FINAL TARGET MOTION ACCUMULATION FOR ACTION-RATE REWARD
            # ============================================================
            # target_step is the actual reference motion produced by the
            # velocity/acceleration limiter at the physics rate (200 Hz).
            # This is what we want action_rate to measure.
            self._target_rate_sq_sum += torch.square(target_step)

            # First command after reset: synchronize immediately.


            self._rate_limited_target = new_target
            self._rate_limited_target_velocity = new_target_velocity

            self._rate_limited_target_initialized = (
                self._rate_limited_target_initialized
                | first_target
            )

            # This is now the target seen by the position-error deadzone / PD loop.
            servo_position_command = self._rate_limited_target            

            raw_position_error = servo_position_command - joint_pos
            raw_position_error_abs = torch.abs(raw_position_error)


            effective_position_error = (
                torch.sign(raw_position_error)
                * torch.relu(
                    raw_position_error_abs
                    - POSITION_ERROR_DEADBAND_RAD
                )
            )

            position_deadzone_active = (
                raw_position_error_abs
                <= POSITION_ERROR_DEADBAND_RAD
            )

            effective_position_target = joint_pos + effective_position_error


            # ============================================================
            # FOCUSED SERVO DEADBAND DEBUG
            # ============================================================

            self._servo_debug_counter += 1

            if (
                self._servo_debug_counter <= self._servo_debug_max_prints
                and self._servo_debug_counter % self._servo_debug_print_every == 0
            ):
                env_id = 0
                idx = self._debug_joint_index

                candidate_deg = (
                    delayed_position[env_id, idx]
                    * 180.0 / math.pi
                )

                previous_deg = (
                    previous_servo_command[env_id, idx]
                    * 180.0 / math.pi
                )

                delta_deg = (
                    command_delta[env_id, idx]
                    * 180.0 / math.pi
                )

                accepted_deg = (
                    self._last_servo_command[env_id, idx]
                    * 180.0 / math.pi
                )

                accepted_change_deg = (
                    self._last_servo_command[env_id, idx]
                    - previous_servo_command[env_id, idx]
                ) * 180.0 / math.pi

                q_real_deg = (
                    joint_pos[env_id, idx]
                    * 180.0 / math.pi
                )

                qd_real_deg_s = (
                    joint_vel[env_id, idx]
                    * 180.0 / math.pi
                )

                q_error_deg = (
                    raw_position_error[env_id, idx]
                ) * 180.0 / math.pi

                effective_q_error_deg = (
                    effective_position_error[env_id, idx]
                ) * 180.0 / math.pi

                effective_target_deg = (
                    effective_position_target[env_id, idx]
                    * 180.0 / math.pi
                )

                deadzone_active = bool(
                    position_deadzone_active[env_id, idx]
                )

                print(
                    "\n================ SERVO DEADBAND TEST ================\n"
                    f"step              = {self._servo_debug_counter}\n"
                    f"joint             = {self.joint_names[idx]}\n"
                    f"candidate         = {candidate_deg:+8.3f}°\n"
                    f"previous accepted = {previous_deg:+8.3f}°\n"
                    f"command delta     = {delta_deg:+8.3f}°\n"
                    f"blocked           = {bool(self._servo_command_blocked[env_id, idx])}\n"
                    f"accepted command  = {accepted_deg:+8.3f}°\n"
                    f"accepted change   = {accepted_change_deg:+8.3f}°\n"
                    f"REAL joint q      = {q_real_deg:+8.3f}°\n"
                    f"REAL joint vel    = {qd_real_deg_s:+8.3f}°/s\n"
                    f"RAW position error = {q_error_deg:+8.3f}°\n"
                    f"deadzone active    = {deadzone_active}\n"
                    f"effective q error  = {effective_q_error_deg:+8.3f}°\n"
                    f"effective target   = {effective_target_deg:+8.3f}°\n"
                    "=====================================================\n",
                    flush=True,
                )






            # Backlash is applied only after the position-error deadzone.
            control_action.joint_positions = self._apply_backlash(
                effective_position_target
            )

        # ================================================================
        # 3. DELAY OTHER COMMAND TYPES
        # ================================================================
        if control_action.joint_velocities is not None:
            control_action.joint_velocities = (
                self.velocities_delay_buffer.compute(
                    control_action.joint_velocities
                )
            )

        if control_action.joint_efforts is not None:
            control_action.joint_efforts = (
                self.efforts_delay_buffer.compute(
                    control_action.joint_efforts
                )
            )

        debug_q_des = (
            control_action.joint_positions.clone()
            if control_action.joint_positions is not None
            else None
        )

        debug_qd_des = (
            control_action.joint_velocities.clone()
            if control_action.joint_velocities is not None
            else None
        )

        debug_tau_ff = (
            control_action.joint_efforts.clone()
            if control_action.joint_efforts is not None
            else None
        )

        # ================================================================
        # 4. LET ISAAC LAB CALCULATE THE REAL ACTUATOR TORQUE
        # ================================================================

        result = super().compute(
            control_action,
            joint_pos,
            joint_vel,
        )

        # ================================================================
        # 4. FULL CONTROL-LOOP TRACE
        # ================================================================
        # IMPORTANT: q/qd are the state BEFORE the torque computed above
        # is applied to the next physics step. Therefore actual acceleration
        # is paired with the PREVIOUS applied torque, while current torque is
        # the controller output for the next step.

        env_id = 0
        rad_to_deg = 180.0 / math.pi
        trace_dt = PHYSICS_DT

        # Current physical states.
        q_all = joint_pos
        qd_all = joint_vel

        # Current desired states (after actuator-side delay/backlash).
        q_des_all = (
            debug_q_des
            if debug_q_des is not None
            else torch.zeros_like(q_all)
        )
        qd_des_all = (
            debug_qd_des
            if debug_qd_des is not None
            else torch.zeros_like(qd_all)
        )

        tau_ff_all = (
            debug_tau_ff
            if debug_tau_ff is not None
            else torch.zeros_like(q_all)
        )

        kp_all = self.stiffness
        kd_all = self.damping
        applied_tau_all = self.applied_effort
        computed_tau_all = self.computed_effort
        final_tau_all = result.joint_efforts
        if final_tau_all is None:
            final_tau_all = applied_tau_all

        position_error_all = q_des_all - q_all
        velocity_error_all = qd_des_all - qd_all

        p_term_all = kp_all * position_error_all
        d_term_all = kd_all * velocity_error_all
        raw_pd_tau_all = p_term_all + d_term_all + tau_ff_all

        effective_effort_limit_all = (
            self.effort_limit * self.torque_scale
        )
        effective_saturation_all = (
            self._saturation_effort * self.torque_scale
        )
        effective_velocity_limit_all = (
            self.velocity_limit * self.speed_scale
        ).clamp_min(1.0e-6)

        # Reconstruct the same torque-speed envelope used by _clip_effort.
        allowed_max_all = torch.clip(
            effective_saturation_all
            * (1.0 - qd_all / effective_velocity_limit_all),
            min=self._zeros_effort,
            max=effective_effort_limit_all,
        )
        allowed_min_all = torch.clip(
            effective_saturation_all
            * (-1.0 - qd_all / effective_velocity_limit_all),
            min=-effective_effort_limit_all,
            max=self._zeros_effort,
        )

        # Quantify actual motor demand against the instantaneous directional
        # torque-speed envelope. Accumulate at 200 Hz; policy reward consumes
        # its four-sample buffer at 50 Hz.
        directional_torque_limit = torch.where(
            raw_pd_tau_all >= 0.0, allowed_max_all, -allowed_min_all
        ).clamp_min(1.0e-6)
        torque_util = torch.nan_to_num(
            torch.abs(final_tau_all) / directional_torque_limit,
            nan=0.0, posinf=2.0, neginf=0.0,
        ).clamp_(0.0, 2.0)
        speed_util = torch.nan_to_num(
            torch.abs(qd_all) / effective_velocity_limit_all,
            nan=0.0, posinf=2.0, neginf=0.0,
        ).clamp_(0.0, 2.0)
        clip_deficit = torch.nan_to_num(
            torch.abs(raw_pd_tau_all - final_tau_all) / directional_torque_limit,
            nan=0.0, posinf=2.0, neginf=0.0,
        ).clamp_(0.0, 2.0)
        saturation_fraction = (torque_util >= 0.95).to(dtype=torch.float32)
        motor_state = self._robotdog_motor_telemetry
        telemetry_order = self._robotdog_telemetry_joint_indices
        # The actuator's control tensors stay in native resolved order. Only
        # the diagnostic/competence telemetry is permuted into numeric joint
        # order, so joint triplets map correctly to legs downstream.
        torque_util_canonical = torque_util.index_select(1, telemetry_order)
        speed_util_canonical = speed_util.index_select(1, telemetry_order)
        clip_deficit_canonical = clip_deficit.index_select(1, telemetry_order)
        saturation_canonical = saturation_fraction.index_select(1, telemetry_order)
        motor_state["torque_util_sum"] += torque_util_canonical
        motor_state["speed_util_sum"] += speed_util_canonical
        motor_state["clip_deficit_sum"] += clip_deficit_canonical
        motor_state["saturation_sum"] += saturation_canonical
        motor_state["episode_samples"] += 1.0
        motor_state["policy_torque_util_sum"] += torque_util_canonical
        motor_state["policy_clip_deficit_sum"] += clip_deficit_canonical
        motor_state["policy_saturation_sum"] += saturation_canonical
        motor_state["policy_samples"] += 1.0

        self._trace_counter += 1

        # First sample initializes the history; afterwards finite differences
        # are meaningful.
        if not torch.all(self._trace_initialized):
            uninitialized = ~self._trace_initialized
            self._trace_prev_q[uninitialized] = q_all[uninitialized]
            self._trace_prev_qd[uninitialized] = qd_all[uninitialized]
            self._trace_prev_target[uninitialized] = q_des_all[uninitialized]
            self._trace_prev_applied_tau[uninitialized] = applied_tau_all[uninitialized]
            self._trace_target_age_steps[uninitialized] = 0
            self._trace_motion_age_steps[uninitialized] = 0
            self._trace_initialized[uninitialized] = True

        delta_q_all = q_all - self._trace_prev_q
        delta_qd_all = qd_all - self._trace_prev_qd
        delta_target_all = q_des_all - self._trace_prev_target

        # Acceleration estimated from the physical velocity history.
        accel_all = delta_qd_all / trace_dt

        target_changed = torch.abs(delta_target_all) >= self._trace_target_threshold_rad
        moving_now = torch.abs(qd_all) >= self._trace_motion_velocity_threshold_rad_s

        self._trace_target_age_steps = torch.where(
            target_changed,
            torch.zeros_like(self._trace_target_age_steps),
            self._trace_target_age_steps + 1,
        )
        self._trace_motion_age_steps = torch.where(
            moving_now,
            torch.zeros_like(self._trace_motion_age_steps),
            self._trace_motion_age_steps + 1,
        )

        if (
            self._trace_counter <= self._trace_max_prints
            and self._trace_counter % self._trace_print_every == 0
        ):
            print(
                "\n"
                "================ CONTROL-LOOP TRACE =================\n"
                f"physics_step: {self._trace_counter} | dt={trace_dt*1000.0:.2f} ms | "
                f"policy_dt={POLICY_DT*1000.0:.2f} ms\n"
            )

            for debug_idx, joint_name in enumerate(self.joint_names):
                q = q_all[:, debug_idx]
                qd = qd_all[:, debug_idx]
                q_des = q_des_all[:, debug_idx]
                qd_des = qd_des_all[:, debug_idx]
                tau_ff = tau_ff_all[:, debug_idx]
                position_error = position_error_all[:, debug_idx]
                velocity_error = velocity_error_all[:, debug_idx]
                p_term = p_term_all[:, debug_idx]
                d_term = d_term_all[:, debug_idx]
                raw_pd_tau = raw_pd_tau_all[:, debug_idx]
                computed_tau = computed_tau_all[:, debug_idx]
                applied_tau = applied_tau_all[:, debug_idx]
                prev_applied_tau = self._trace_prev_applied_tau[:, debug_idx]
                dq = delta_q_all[:, debug_idx]
                dtarget = delta_target_all[:, debug_idx]
                accel = accel_all[:, debug_idx]
                allowed_max = allowed_max_all[:, debug_idx]
                allowed_min = allowed_min_all[:, debug_idx]

                effective_effort_limit = effective_effort_limit_all[:, debug_idx]
                command_delta_deg = (
                    self._servo_command_requested_delta[:, debug_idx]
                    * rad_to_deg
                )
                command_blocked = self._servo_command_blocked[:, debug_idx]
                saturation_ratio = (
                    torch.abs(applied_tau[env_id])
                    / torch.clamp(effective_effort_limit[env_id], min=1.0e-9)
                )

                # Direction flags help identify who reversed first.
                # Target reversal is not used as the primary event classifier
                # because a single finite difference is not enough to establish
                # a sustained reversal. The raw dT is still printed every trace.

                vel_reversed = (
                    torch.abs(qd[env_id]) >= self._trace_motion_velocity_threshold_rad_s
                    and torch.abs(self._trace_prev_qd[env_id, debug_idx])
                    >= self._trace_motion_velocity_threshold_rad_s
                    and torch.sign(qd[env_id])
                    != torch.sign(self._trace_prev_qd[env_id, debug_idx])
                )

                tau_reversed = (
                    torch.abs(applied_tau[env_id]) >= 0.05
                    and torch.abs(prev_applied_tau[env_id]) >= 0.05
                    and torch.sign(applied_tau[env_id])
                    != torch.sign(prev_applied_tau[env_id])
                )

                target_static = (
                    self._trace_target_age_steps[env_id, debug_idx].item()
                    * trace_dt
                    >= 3.0 * trace_dt
                )

                if target_static and vel_reversed:
                    event = "OSCILLATION? target_static + vel_reversed"
                elif target_static and tau_reversed:
                    event = "TORQUE_REVERSED target_static"
                elif target_changed[env_id, debug_idx]:
                    event = "TARGET_CHANGED"
                elif saturation_ratio >= 0.95:
                    event = "TORQUE_SATURATED"
                else:
                    event = "-"

                print(
                    f"{joint_name:12s} | "
                    f"q={q[env_id].item()*rad_to_deg:+7.2f}° | "
                    f"target={q_des[env_id].item()*rad_to_deg:+7.2f}° | "
                    f"dT={dtarget[env_id].item()*rad_to_deg:+6.2f}° | "
                    f"dQ={dq[env_id].item()*rad_to_deg:+6.2f}° | "
                    f"err={position_error[env_id].item()*rad_to_deg:+6.2f}° | "
                    f"vel={qd[env_id].item()*rad_to_deg:+8.2f}°/s | "
                    f"acc={accel[env_id].item()*rad_to_deg:+9.1f}°/s² | "
                    f"tT={self._trace_target_age_steps[env_id, debug_idx].item()*trace_dt:5.3f}s | "
                    f"tM={self._trace_motion_age_steps[env_id, debug_idx].item()*trace_dt:5.3f}s | "
                    f"P={p_term[env_id].item():+6.3f} | "
                    f"D={d_term[env_id].item():+6.3f} | "
                    f"PD={raw_pd_tau[env_id].item():+6.3f} Nm | "
                    f"calc={computed_tau[env_id].item():+6.3f} | "
                    f"tau={applied_tau[env_id].item():+6.3f} | "
                    f"prev_tau={prev_applied_tau[env_id].item():+6.3f} | "
                    f"lim=[{allowed_min[env_id].item():+.2f},{allowed_max[env_id].item():+.2f}] | "
                    f"sat={saturation_ratio.item():.2f} | "
                    f"final={final_tau_all[env_id, debug_idx].item():+7.3f} | "
                    f"cmd_d={command_delta_deg[env_id].item():+6.3f}° | "
                    f"blocked={int(command_blocked[env_id].item())} | "
                    # f"settled={debug_settled[env_id, debug_idx].item()} | "
                    f"{event}"
                )

            print(
                "======================================================\n",
                flush=True,
            )

        # Update history AFTER printing so prev_* refers to the previous
        # physical/control sample on the next call.
        self._trace_prev_q.copy_(q_all)
        self._trace_prev_qd.copy_(qd_all)
        self._trace_prev_target.copy_(q_des_all)
        self._trace_prev_applied_tau.copy_(applied_tau_all)

        return result






@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """DCMotor actuator with randomized command delay."""

    class_type: type = DelayedDCMotor

    min_delay: int = 0
    max_delay: int = 0

    # Target trajectory limits.
    target_velocity_limit: float = JOINT_TARGET_MAX_VELOCITY_RAD_S
    target_acceleration_limit: float = JOINT_TARGET_MAX_ACCELERATION_RAD_S2

    # Backlash parameters.
    backlash_min: float = 0.0
    backlash_max: float = 0.0

    # Per-environment / per-joint actuator variation.
    torque_scale_min: float = SERVO_TORQUE_SCALE_MIN
    torque_scale_max: float = SERVO_TORQUE_SCALE_MAX

    speed_scale_min: float = SERVO_SPEED_SCALE_MIN
    speed_scale_max: float = SERVO_SPEED_SCALE_MAX

    # Per-environment / per-joint joint zero calibration error.
    zero_offset_min_rad: float = SERVO_ZERO_OFFSET_MIN_RAD
    zero_offset_max_rad: float = SERVO_ZERO_OFFSET_MAX_RAD



   

@configclass
class RobotdogSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_GENERATOR_CFG,
        # Initial origins start at row zero. LP-ACRL assigns the first episode
        # tasks before Scene.reset(), using a warm-start distribution over rows.
        max_init_terrain_level=0,
        env_spacing=3.0,
        collision_group=-1,

        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),

        debug_vis=False,
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",

        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            # REZOLVARE: Adaugă acest bloc pentru a forța activarea coliziunilor la nivel de spawner
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.180),
            joint_pos=DEFAULT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),

        actuators={
            "all_joints": DelayedDCMotorCfg(
                joint_names_expr=[".*"],

                # PD controller.
                stiffness=SERVO_KP,
                damping=SERVO_KD,

                target_velocity_limit=JOINT_TARGET_MAX_VELOCITY_RAD_S,
                target_acceleration_limit=JOINT_TARGET_MAX_ACCELERATION_RAD_S2,

                # Friction / inertia.
                friction=0.08,
                dynamic_friction=0.04,
                viscous_friction=0.015,
                armature=0.005,

                # STS3215 model. The supply selector above must match the measured servo rail.
                saturation_effort=STS3215_PEAK_TORQUE_NM,
                effort_limit=STS3215_RATED_TORQUE_NM,
                velocity_limit=STS3215_NO_LOAD_SPEED,

                # Physics solver limits.
                effort_limit_sim=STS3215_PEAK_TORQUE_NM,
                velocity_limit_sim=STS3215_NO_LOAD_SPEED,

                # Actuator delay.
                min_delay=ACTUATOR_DELAY_MIN,
                max_delay=ACTUATOR_DELAY_MAX,
                
                # Actuator backlash
                backlash_min=BACKLASH_MIN_RAD,
                backlash_max=BACKLASH_MAX_RAD,

                speed_scale_min=SERVO_SPEED_SCALE_MIN,
                speed_scale_max=SERVO_SPEED_SCALE_MAX,

                zero_offset_min_rad=SERVO_ZERO_OFFSET_MIN_RAD,
                zero_offset_max_rad=SERVO_ZERO_OFFSET_MAX_RAD,
            ),
        },
    )   


     # ---------------------------------------------------------------------
    # CONTACT SENSORS
    # ---------------------------------------------------------------------
    # IMPORTANT: Dacă îți dă eroare, înseamnă că numele link-urilor din USD
    # nu se potrivesc cu pattern-urile de mai jos.
    # Spune-mi exact eroarea + lista de body names și le corectăm.
    
    # ---------------------------------------------------------------------
    # IMU
    # ---------------------------------------------------------------------
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1",
        update_period=1.0 / 120.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        debug_vis=False,
    )



    servo3_contact_1 = ContactSensorCfg(
        update_period=PHYSICS_DT,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_1__body1/servo_3_1__body1",
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        history_length=1,
        track_air_time=True,
        force_threshold=1.0,
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
        debug_vis=False,
    )

    servo3_contact_2 = ContactSensorCfg(
        update_period=PHYSICS_DT,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_mirror__1__body1/servo_3_mirror__1__body1",
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        history_length=1,
        track_air_time=True,
        force_threshold=1.0,
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
        debug_vis=False,
    )

    servo3_contact_3 = ContactSensorCfg(
        update_period=PHYSICS_DT,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_mirror___1__1__body1/servo_3_mirror___1__1__body1",
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        history_length=1,
        track_air_time=True,
        force_threshold=1.0,
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
        debug_vis=False,
    )

    servo3_contact_4 = ContactSensorCfg(
        update_period=PHYSICS_DT,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2__1__1__body1/servo_3__1__1__body1",
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        history_length=1,
        track_air_time=True,
        force_threshold=1.0,
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
        debug_vis=False,
    )


    # Keep DECIMATION physics samples so ground-filtered foot timers can be
    # reconstructed at PHYSICS_DT without relying on unfiltered sensor timers.
    foot_1 = ContactSensorCfg(
        update_period=PHYSICS_DT,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_1__body1/servo_3_1__body1/componentfinal_1__body1",
        history_length=DECIMATION,
        track_air_time=True,
        force_threshold=1.0,
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
        debug_vis=False,
    )

    foot_2 = ContactSensorCfg(
        update_period=PHYSICS_DT,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_mirror__1__body1/servo_3_mirror__1__body1/componentfinal_mirror__1__body1",
        history_length=DECIMATION,
        track_air_time=True,
        force_threshold=1.0,
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
        debug_vis=False,
    )

    foot_3 = ContactSensorCfg(
        update_period=PHYSICS_DT,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_mirror___1__1__body1/servo_3_mirror___1__1__body1/componentfinal_mirror___1__1__body1",
        history_length=DECIMATION,
        track_air_time=True,
        force_threshold=1.0,
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
        debug_vis=False,
    )

    foot_4 = ContactSensorCfg(
        update_period=PHYSICS_DT,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2__1__1__body1/servo_3__1__1__body1/componentfinal__1__1__body1",
        history_length=DECIMATION,
        track_air_time=True,
        force_threshold=1.0,
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
        debug_vis=False,
    )

    # Terrain ray samples follow the moving foot body but remain gravity-aligned.
    # The support-point query below compares the actual USD collision surface
    # against the nearest valid terrain hit, not against the link origin.
    foot_1_clearance_rays = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_1__body1/servo_3_1__body1/componentfinal_1__body1",
        update_period=POLICY_DT,
        offset=RayCasterCfg.OffsetCfg(pos=(FOOT_SOLE_RAY_CENTER_LOCAL_M[0][0], FOOT_SOLE_RAY_CENTER_LOCAL_M[0][1], FOOT_SOLE_RAY_CENTER_LOCAL_M[0][2] + FOOT_CLEARANCE_RAY_ORIGIN_HEIGHT_M)),
        mesh_prim_paths=["/World/ground/terrain/mesh"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=FOOT_CLEARANCE_RAY_RESOLUTION_M,
            size=FOOT_CLEARANCE_RAY_GRID_SIZE_M,
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=FOOT_CLEARANCE_RAY_MAX_DISTANCE_M,
        debug_vis=False,
    )

    # Terrain ray samples follow the moving foot body but remain gravity-aligned.
    # The support-point query below compares the actual USD collision surface
    # against the nearest valid terrain hit, not against the link origin.
    foot_2_clearance_rays = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_mirror__1__body1/servo_3_mirror__1__body1/componentfinal_mirror__1__body1",
        update_period=POLICY_DT,
        offset=RayCasterCfg.OffsetCfg(pos=(FOOT_SOLE_RAY_CENTER_LOCAL_M[1][0], FOOT_SOLE_RAY_CENTER_LOCAL_M[1][1], FOOT_SOLE_RAY_CENTER_LOCAL_M[1][2] + FOOT_CLEARANCE_RAY_ORIGIN_HEIGHT_M)),
        mesh_prim_paths=["/World/ground/terrain/mesh"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=FOOT_CLEARANCE_RAY_RESOLUTION_M,
            size=FOOT_CLEARANCE_RAY_GRID_SIZE_M,
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=FOOT_CLEARANCE_RAY_MAX_DISTANCE_M,
        debug_vis=False,
    )

    # Terrain ray samples follow the moving foot body but remain gravity-aligned.
    # The support-point query below compares the actual USD collision surface
    # against the nearest valid terrain hit, not against the link origin.
    foot_3_clearance_rays = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2_mirror___1__1__body1/servo_3_mirror___1__1__body1/componentfinal_mirror___1__1__body1",
        update_period=POLICY_DT,
        offset=RayCasterCfg.OffsetCfg(pos=(FOOT_SOLE_RAY_CENTER_LOCAL_M[2][0], FOOT_SOLE_RAY_CENTER_LOCAL_M[2][1], FOOT_SOLE_RAY_CENTER_LOCAL_M[2][2] + FOOT_CLEARANCE_RAY_ORIGIN_HEIGHT_M)),
        mesh_prim_paths=["/World/ground/terrain/mesh"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=FOOT_CLEARANCE_RAY_RESOLUTION_M,
            size=FOOT_CLEARANCE_RAY_GRID_SIZE_M,
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=FOOT_CLEARANCE_RAY_MAX_DISTANCE_M,
        debug_vis=False,
    )

    # Terrain ray samples follow the moving foot body but remain gravity-aligned.
    # The support-point query below compares the actual USD collision surface
    # against the nearest valid terrain hit, not against the link origin.
    foot_4_clearance_rays = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1/servo2__1__1__body1/servo_3__1__1__body1/componentfinal__1__1__body1",
        update_period=POLICY_DT,
        offset=RayCasterCfg.OffsetCfg(pos=(FOOT_SOLE_RAY_CENTER_LOCAL_M[3][0], FOOT_SOLE_RAY_CENTER_LOCAL_M[3][1], FOOT_SOLE_RAY_CENTER_LOCAL_M[3][2] + FOOT_CLEARANCE_RAY_ORIGIN_HEIGHT_M)),
        mesh_prim_paths=["/World/ground/terrain/mesh"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=FOOT_CLEARANCE_RAY_RESOLUTION_M,
            size=FOOT_CLEARANCE_RAY_GRID_SIZE_M,
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=FOOT_CLEARANCE_RAY_MAX_DISTANCE_M,
        debug_vis=False,
    )

    # Small downward support-height probe beneath the body. This is used only
    # by the base-height reward so stepping onto a raised platform does not
    # incorrectly look like an 8 cm posture error relative to the tile origin.
    base_ground_scan = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1",
        update_period=POLICY_DT,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.45)),
        mesh_prim_paths=["/World/ground/terrain/mesh"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.08,
            size=(0.16, 0.16),
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=1.0,
        debug_vis=False,
    )

    # Teacher-only anticipatory terrain scan.  The sensor is attached to the
    # base, but the ray grid is shifted forward along physical local -X.
    # 17 x 33 samples = 561 rays per environment; the wide lateral field of
    # view is required to observe both alternating bypass openings.
    terrain_lookahead = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1",
        update_period=POLICY_DT,
        offset=RayCasterCfg.OffsetCfg(
            pos=(TERRAIN_LOOKAHEAD_OFFSET_X_M, 0.0, TERRAIN_LOOKAHEAD_OFFSET_Z_M)
        ),
        mesh_prim_paths=["/World/ground/terrain/mesh"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=TERRAIN_LOOKAHEAD_RESOLUTION_M,
            size=TERRAIN_LOOKAHEAD_SIZE_M,
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=1.0,
        debug_vis=False,
    )

    base_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/root/base_link__2__1__body1",
        update_period=0.0,
        max_contact_data_count_per_prim=8,  # detailed-contact buffer capacity; not a force-matrix contact threshold
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=[
            "/World/ground/terrain/mesh"
        ],
    )

    dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.9, 0.9, 0.9),
                intensity=500.0,
            ),
        )


# =============================================================================
# TEACHER CONTROLS + ORACLE TERRAIN MAP (NO LIDAR / NO GAIT REFERENCE)
# =============================================================================

# The procedural terrain tiles are 8 x 8 m. The map is sampled at the same
# horizontal resolution as the generated terrain mesh and follows the robot.
# It is not generated by virtual rays: it is rasterized directly from the
# ground-truth triangle mesh created by TerrainImporter.
TERRAIN_MAP_SIZE_M = tuple(float(v) for v in TERRAIN_GENERATOR_CFG.size)
TERRAIN_MAP_RESOLUTION_M = float(getattr(TERRAIN_GENERATOR_CFG, "horizontal_scale", 0.10))
TERRAIN_MAP_SAMPLES_X = int(round(TERRAIN_MAP_SIZE_M[0] / TERRAIN_MAP_RESOLUTION_M)) + 1
TERRAIN_MAP_SAMPLES_Y = int(round(TERRAIN_MAP_SIZE_M[1] / TERRAIN_MAP_RESOLUTION_M)) + 1
TERRAIN_MAP_FEATURES = TERRAIN_MAP_SAMPLES_X * TERRAIN_MAP_SAMPLES_Y

# Rezoluția hărții globale de ansamblu.
# Harta locală de 8x8 m rămâne la rezoluția de 10 cm.
GLOBAL_TERRAIN_OVERVIEW_RESOLUTION_M = 2.0


def _get_oracle_terrain_vertices(env):
    """Read the generated terrain mesh directly from the active USD stage."""

    import omni.usd
    from pxr import Usd, UsdGeom

    print(
        "[RobotDog oracle map] Reading terrain geometry from USD stage...",
        flush=True,
    )

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError(
            "[RobotDog oracle map] No active USD stage is available."
        )

    terrain = env.scene["terrain"]

    # TerrainImporter exposes the actual imported terrain prim paths.
    terrain_paths = list(
        getattr(terrain, "terrain_prim_paths", [])
    )

    if not terrain_paths:
        terrain_paths = ["/World/ground/terrain"]

    vertex_chunks = []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    for root_path in terrain_paths:
        root_prim = stage.GetPrimAtPath(root_path)

        if not root_prim.IsValid():
            print(
                f"[RobotDog oracle map] Invalid terrain prim: {root_path}",
                flush=True,
            )
            continue

        # The generated TerrainImporter mesh is stored under the terrain
        # prim. Search descendants instead of using the deprecated .meshes.
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue

            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()

            if points is None or len(points) == 0:
                continue

            vertices = np.asarray(points, dtype=np.float32)

            if vertices.ndim != 2 or vertices.shape[1] < 3:
                raise RuntimeError(
                    "[RobotDog oracle map] Unexpected USD mesh point shape "
                    f"at {prim.GetPath()}: {vertices.shape}"
                )

            vertices = vertices[:, :3]

            # Apply the USD mesh's local-to-world transform.
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            matrix_np = np.asarray(
                [
                    [matrix[r][c] for c in range(4)]
                    for r in range(4)
                ],
                dtype=np.float64,
            )

            if not np.allclose(
                matrix_np, np.eye(4), atol=1.0e-9
            ):
                homogeneous = np.ones(
                    (vertices.shape[0], 4), dtype=np.float64
                )
                homogeneous[:, :3] = vertices

                vertices = (
                    homogeneous @ matrix_np
                )[:, :3].astype(np.float32)

            vertex_chunks.append(vertices)

            print(
                f"[RobotDog oracle map] Found mesh "
                f"{prim.GetPath()}: "
                f"{vertices.shape[0]:,} vertices",
                flush=True,
            )

    if not vertex_chunks:
        raise RuntimeError(
            "[RobotDog oracle map] No UsdGeom.Mesh vertices were found "
            f"under terrain paths: {terrain_paths!r}. "
            "The USD mesh path must be inspected."
        )

    vertices = np.concatenate(vertex_chunks, axis=0)

    vertices = vertices[
        np.isfinite(vertices[:, :3]).all(axis=1)
    ]

    if vertices.shape[0] == 0:
        raise RuntimeError(
            "[RobotDog oracle map] No finite terrain vertices were found."
        )

    print(
        f"[RobotDog oracle map] Loaded {vertices.shape[0]:,} "
        "finite vertices from USD.",
        flush=True,
    )

    return vertices


def _get_oracle_height_grid(env):
    """Rasterize the real generated terrain mesh once; cache the grid on the env."""
    cache_name = "_robotdog_oracle_global_height_grid"
    if hasattr(env, cache_name):
        return getattr(env, cache_name)

    vertices = _get_oracle_terrain_vertices(env)
    if vertices.ndim != 2 or vertices.shape[1] < 3 or vertices.shape[0] == 0:
        raise RuntimeError(
            f"[RobotDog oracle map] Invalid terrain-mesh vertices: shape={vertices.shape}."
        )

    finite_vertices = np.isfinite(vertices[:, :3]).all(axis=1)
    vertices = vertices[finite_vertices, :3]
    if vertices.shape[0] == 0:
        raise RuntimeError("[RobotDog oracle map] Terrain mesh has no finite vertices.")

    resolution = TERRAIN_MAP_RESOLUTION_M
    x_min = float(vertices[:, 0].min())
    y_min = float(vertices[:, 1].min())
    x_max = float(vertices[:, 0].max())
    y_max = float(vertices[:, 1].max())
    num_x = int(round((x_max - x_min) / resolution)) + 1
    num_y = int(round((y_max - y_min) / resolution)) + 1

    # The generated terrain is a height-field terrain: at each XY coordinate,
    # use the highest mesh vertex. This also gives obstacle tops / stair tops,
    # rather than a lower vertical-riser vertex at duplicated XY coordinates.
    ix = np.rint((vertices[:, 0] - x_min) / resolution).astype(np.int64)
    iy = np.rint((vertices[:, 1] - y_min) / resolution).astype(np.int64)
    in_bounds = (ix >= 0) & (ix < num_x) & (iy >= 0) & (iy < num_y)
    ix = ix[in_bounds]
    iy = iy[in_bounds]
    vz = vertices[in_bounds, 2]

    height_grid_np = np.full((num_x, num_y), -np.inf, dtype=np.float32)
    np.maximum.at(height_grid_np, (ix, iy), vz)
    validity_np = np.isfinite(height_grid_np) & (height_grid_np > -np.inf)
    height_grid_np[~validity_np] = 0.0

    height_grid = torch.as_tensor(height_grid_np, dtype=torch.float32, device=env.device)
    validity_grid = torch.as_tensor(validity_np, dtype=torch.bool, device=env.device)

    cache = {
        "height_grid": height_grid,
        "validity_grid": validity_grid,
        "x_min": x_min,
        "y_min": y_min,
        "resolution": resolution,
        "num_x": num_x,
        "num_y": num_y,
        "mesh_vertex_count": int(vertices.shape[0]),
        "valid_grid_cells": int(validity_np.sum()),
        "total_grid_cells": int(validity_np.size),
    }
    setattr(env, cache_name, cache)

    coverage = cache["valid_grid_cells"] / max(1, cache["total_grid_cells"])
    print(
        "[RobotDog oracle map] Rasterized terrain mesh directly: "
        f"vertices={cache['mesh_vertex_count']:,}; grid={num_x}x{num_y}; "
        f"sample coverage={coverage:.2%}; map window="
        f"{TERRAIN_MAP_SIZE_M[0]:.1f}x{TERRAIN_MAP_SIZE_M[1]:.1f} m @ {resolution:.3f} m.",
        flush=True,
    )
    return cache


def _teacher_oracle_map_samples(env):
    """Sample a robot-centred, robot-yaw-aligned full terrain patch from the oracle grid."""
    cache = _get_oracle_height_grid(env)
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w
    quat = robot.data.root_quat_w  # Isaac Lab quaternion order: (w, x, y, z).
    qw, qx, qy, qz = quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c = torch.cos(yaw).view(-1, 1, 1)
    s = torch.sin(yaw).view(-1, 1, 1)

    x_local = torch.linspace(
        -TERRAIN_MAP_SIZE_M[0] / 2.0,
        TERRAIN_MAP_SIZE_M[0] / 2.0,
        TERRAIN_MAP_SAMPLES_X,
        dtype=root_pos.dtype,
        device=env.device,
    )
    y_local = torch.linspace(
        -TERRAIN_MAP_SIZE_M[1] / 2.0,
        TERRAIN_MAP_SIZE_M[1] / 2.0,
        TERRAIN_MAP_SAMPLES_Y,
        dtype=root_pos.dtype,
        device=env.device,
    )
    grid_x_local, grid_y_local = torch.meshgrid(x_local, y_local, indexing="ij")
    grid_x_local = grid_x_local.unsqueeze(0)
    grid_y_local = grid_y_local.unsqueeze(0)

    world_x = root_pos[:, 0].view(-1, 1, 1) + c * grid_x_local - s * grid_y_local
    world_y = root_pos[:, 1].view(-1, 1, 1) + s * grid_x_local + c * grid_y_local

    ix_float = (world_x - cache["x_min"]) / cache["resolution"]
    iy_float = (world_y - cache["y_min"]) / cache["resolution"]
    ix = torch.round(ix_float).to(torch.long)
    iy = torch.round(iy_float).to(torch.long)

    in_bounds = (
        (ix >= 0) & (ix < cache["num_x"])
        & (iy >= 0) & (iy < cache["num_y"])
    )
    ix_safe = ix.clamp(0, cache["num_x"] - 1)
    iy_safe = iy.clamp(0, cache["num_y"] - 1)
    linear_indices = ix_safe * cache["num_y"] + iy_safe
    flat_height = cache["height_grid"].reshape(-1)
    flat_valid = cache["validity_grid"].reshape(-1)
    sampled_height = flat_height[linear_indices]
    sampled_valid = flat_valid[linear_indices] & in_bounds

    # Heights are expressed relative to the robot's root height. This tells the
    # Teacher the actual ground elevation around its body without any ray sensor.
    relative_height = sampled_height - root_pos[:, 2].view(-1, 1, 1)
    relative_height = torch.where(sampled_valid, relative_height, torch.zeros_like(relative_height))
    return relative_height, sampled_valid.to(torch.float32)


def teacher_full_terrain_map(env):
    """Full 8 x 8 m terrain height map from the simulator's true mesh, no raycasting."""
    heights, _ = _teacher_oracle_map_samples(env)
    return heights.reshape(env.num_envs, -1)


def teacher_full_terrain_map_validity(env):
    """Indicates map cells backed by an actual vertex in the generated terrain mesh."""
    _, validity = _teacher_oracle_map_samples(env)
    return validity.reshape(env.num_envs, -1)



def _get_oracle_global_terrain_overview(env):
    """Construiește și memorează o vedere globală, redusă ca rezoluție."""

    cache_name = "_robotdog_oracle_global_terrain_overview_cache"

    if hasattr(env, cache_name):
        return getattr(env, cache_name)

    # Grila globală existentă, extrasă din mesh-ul real al terenului.
    cache = _get_oracle_height_grid(env)

    heights = cache["height_grid"]
    validity = cache["validity_grid"].to(dtype=heights.dtype)

    # Dimensiunile reale ale terenului din mesh, în metri.
    extent_x = (cache["num_x"] - 1) * cache["resolution"]
    extent_y = (cache["num_y"] - 1) * cache["resolution"]

    # Grilă globală cu eșantionare de aproximativ 2 metri.
    out_x = max(
        2,
        int(round(extent_x / GLOBAL_TERRAIN_OVERVIEW_RESOLUTION_M)) + 1,
    )
    out_y = max(
        2,
        int(round(extent_y / GLOBAL_TERRAIN_OVERVIEW_RESOLUTION_M)) + 1,
    )
    output_size = (out_x, out_y)

    # Păstrăm doar contribuția celulelor valide atunci când
    # reducem rezoluția hărții.
    heights_4d = (heights * validity)[None, None, :, :]
    validity_4d = validity[None, None, :, :]

    height_sum = torch.nn.functional.adaptive_avg_pool2d(
        heights_4d, output_size
    )
    valid_fraction = torch.nn.functional.adaptive_avg_pool2d(
        validity_4d, output_size
    )

    coarse_heights = height_sum / valid_fraction.clamp_min(1.0e-6)
    coarse_validity = valid_fraction >= 0.5

    coarse_heights = torch.where(
        coarse_validity,
        coarse_heights,
        torch.zeros_like(coarse_heights),
    )

    x_min = cache["x_min"]
    y_min = cache["y_min"]
    x_max = x_min + extent_x
    y_max = y_min + extent_y

    overview = {
        "height_flat": coarse_heights.reshape(-1),
        "validity_flat": coarse_validity.to(
            dtype=heights.dtype
        ).reshape(-1),
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "shape": (out_x, out_y),
    }

    setattr(env, cache_name, overview)

    print(
        "[RobotDog global oracle] "
        f"Grid={out_x}x{out_y}; "
        f"resolution~{GLOBAL_TERRAIN_OVERVIEW_RESOLUTION_M:.1f} m; "
        f"extent={extent_x:.1f}x{extent_y:.1f} m",
        flush=True,
    )

    return overview


def teacher_global_terrain_map(env):
    """Vedere globală a întregului teren, redusă ca rezoluție."""

    overview = _get_oracle_global_terrain_overview(env)

    return overview["height_flat"].unsqueeze(0).expand(
        env.num_envs, -1
    )


def teacher_global_terrain_map_validity(env):
    """Masca celulelor valide ale hărții globale."""

    overview = _get_oracle_global_terrain_overview(env)

    return overview["validity_flat"].unsqueeze(0).expand(
        env.num_envs, -1
    )


def teacher_global_map_position(env):
    """Poziția XY a robotului, normalizată în coordonatele hărții globale."""

    overview = _get_oracle_global_terrain_overview(env)
    robot = env.scene["robot"]

    position_xy = robot.data.root_pos_w[:, :2]

    span_x = max(overview["x_max"] - overview["x_min"], 1.0e-6)
    span_y = max(overview["y_max"] - overview["y_min"], 1.0e-6)

    x_norm = (
        (position_xy[:, 0] - overview["x_min"]) / span_x
    ) * 2.0 - 1.0

    y_norm = (
        (position_xy[:, 1] - overview["y_min"]) / span_y
    ) * 2.0 - 1.0

    return torch.stack([x_norm, y_norm], dim=-1).clamp(-1.0, 1.0)



def teacher_imu_ang_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")):
    return env.scene[asset_cfg.name].data.ang_vel_b


def teacher_imu_lin_acc(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")):
    return env.scene[asset_cfg.name].data.lin_acc_b


def teacher_projected_gravity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.projected_gravity_b


def _sensor_force_magnitude(env, sensor_name: str) -> torch.Tensor:
    force = env.scene[sensor_name].data.net_forces_w.reshape(env.num_envs, -1, 3)
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=-1)


def teacher_foot_contacts(env):
    """Hysteretic foot-contact states used by event timing and teacher observations."""
    return _robotdog_filtered_foot_contact_timing(env)["contact"].to(torch.float32)


def teacher_body_contact_states(env):
    servo3_contacts = [
        _filtered_terrain_contact_mask(env, name).to(torch.float32)
        for name in SERVO3_SENSOR_NAMES
    ]
    base_contact = (_sensor_force_magnitude(env, "base_contact") > 1.0).to(torch.float32)
    return torch.cat((torch.stack(servo3_contacts, dim=-1), base_contact.unsqueeze(-1)), dim=-1)


def teacher_foot_contact_forces(env):
    return torch.stack(
        [_filtered_terrain_force_magnitude(env, name) for name in FOOT_SENSOR_NAMES],
        dim=-1,
    )


def teacher_base_contact_force(env):
    return _sensor_force_magnitude(env, "base_contact").unsqueeze(-1)


def teacher_base_lin_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.root_lin_vel_b


def teacher_base_ang_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.root_ang_vel_b


def teacher_base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return (env.scene[asset_cfg.name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def robotdog_height_scan(
    env,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.18,
):
    """Legacy helper for older global terrain-map experiments.

    The active V2 teacher uses ``terrain_lookahead_height_scan`` below. This
    older helper is retained for compatibility with dormant map utilities and
    is not registered in the active policy or teacher observation groups.
    """
    sensor = env.scene[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(hit_z)
    scanner_parent_z = sensor.data.pos_w[:, 2].unsqueeze(-1)
    heights = scanner_parent_z - hit_z - offset
    heights = torch.where(valid, heights, torch.zeros_like(heights))

    return heights


def robotdog_height_scan_validity(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """One for a valid terrain return, zero when the ray did not hit terrain."""
    sensor = env.scene[sensor_cfg.name]
    return torch.isfinite(sensor.data.ray_hits_w[..., 2]).to(dtype=torch.float32)


def _terrain_lookahead_sensor_data(env, sensor_cfg: SceneEntityCfg):
    """Fetch and validate the configured forward ray grid."""
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w
    expected_shape = (env.num_envs, TERRAIN_LOOKAHEAD_RAY_COUNT, 3)
    if hits.ndim != 3 or tuple(hits.shape) != expected_shape:
        raise RuntimeError(
            "Terrain lookahead RayCaster returned shape "
            f"{tuple(hits.shape)}; expected {expected_shape}. "
            "Check GridPatternCfg size/resolution and sensor initialization."
        )
    return sensor, hits[..., 2]


def terrain_lookahead_height_scan(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("terrain_lookahead"),
    reference_base_height_m: float = TERRAIN_LOOKAHEAD_REFERENCE_BASE_HEIGHT_M,
    height_scale_m: float = TERRAIN_LOOKAHEAD_HEIGHT_SCALE_M,
):
    """Return normalized teacher-only terrain heights in front of the robot.

    Zero means nominal flat ground at the expected base-to-ground height;
    positive values are raised terrain/obstacles and negative values are dips.
    Heights are divided by the configured 18 cm scale and clipped to [-1, 1].
    This preserves the full range of the 14-16 cm bypass barriers while keeping
    the 2-8 cm climbable surfaces distinguishable. Invalid ray returns are
    zero-filled and separately masked.
    """
    if height_scale_m <= 0.0:
        raise ValueError("height_scale_m must be positive")
    sensor, hit_z = _terrain_lookahead_sensor_data(env, sensor_cfg)
    valid = torch.isfinite(hit_z)
    # `sensor.data.pos_w` is the *offset sensor origin* (base + 42 cm), not
    # the robot root. Normalize against the root height so flat ground at the
    # nominal 18 cm standing height maps to zero.
    root_z = env.scene["robot"].data.root_pos_w[:, 2].unsqueeze(-1)
    nominal_ground_z = root_z - reference_base_height_m
    relative_height = (hit_z - nominal_ground_z) / height_scale_m
    relative_height = torch.clamp(relative_height, min=-1.0, max=1.0)
    return torch.where(valid, relative_height, torch.zeros_like(relative_height))


def terrain_lookahead_validity(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("terrain_lookahead"),
):
    """Return a per-ray validity mask for the forward terrain preview."""
    _, hit_z = _terrain_lookahead_sensor_data(env, sensor_cfg)
    return torch.isfinite(hit_z).to(dtype=torch.float32)


def _terrain_lookahead_noisy_observation(env, sensor_cfg: SceneEntityCfg):
    """One cached noisy elevation scan and validity mask per policy step.

    The ray caster queries static simulator terrain, while this wrapper adds
    small range noise and independent missing returns to avoid training the
    teacher on unrealistically perfect measurements. Height and validity terms
    share the same dropout realization through a step-keyed cache.
    """
    step_key = _robotdog_current_step_key(env)
    cache = getattr(env, "_robotdog_terrain_lookahead_noise_cache", None)
    if cache is not None and cache["step_key"] == step_key:
        return cache["height"], cache["valid"]

    clean_height = terrain_lookahead_height_scan(env, sensor_cfg=sensor_cfg)
    clean_valid = terrain_lookahead_validity(env, sensor_cfg=sensor_cfg) > 0.5
    dropout = torch.rand_like(clean_height) < TERRAIN_LOOKAHEAD_DROPOUT_PROB
    valid = clean_valid & (~dropout)
    noise_std_normalized = TERRAIN_LOOKAHEAD_NOISE_STD_M / TERRAIN_LOOKAHEAD_HEIGHT_SCALE_M
    noisy_height = clean_height + torch.randn_like(clean_height) * noise_std_normalized
    noisy_height = torch.clamp(noisy_height, min=-1.0, max=1.0)
    noisy_height = torch.where(valid, noisy_height, torch.zeros_like(noisy_height))
    valid_float = valid.to(dtype=torch.float32)

    env._robotdog_terrain_lookahead_noise_cache = {
        "step_key": step_key,
        "height": noisy_height,
        "valid": valid_float,
    }
    return noisy_height, valid_float


def terrain_lookahead_noisy_height_scan(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("terrain_lookahead"),
):
    return _terrain_lookahead_noisy_observation(env, sensor_cfg)[0]


def terrain_lookahead_noisy_validity(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("terrain_lookahead"),
):
    return _terrain_lookahead_noisy_observation(env, sensor_cfg)[1]


def _terrain_obstacle_guidance(env):
    """Return (barrier_ahead, preferred_lateral_sign, confidence) from the clean scan.

    The map is X-major/Y-minor. Positive and negative lateral halves are
    compared over the full forward scan so the teacher can select the more
    open side of a staggered wall. Missing returns are excluded from occupancy.
    """
    step_key = _robotdog_current_step_key(env)
    cached = getattr(env, "_robotdog_terrain_obstacle_guidance_cache", None)
    if cached is not None and cached["step_key"] == step_key:
        return cached["barrier"], cached["direction"], cached["confidence"]

    heights_n = terrain_lookahead_height_scan(env)
    valid_n = terrain_lookahead_validity(env) > 0.5
    grid_m = heights_n.reshape(
        env.num_envs, TERRAIN_LOOKAHEAD_SAMPLES_X, TERRAIN_LOOKAHEAD_SAMPLES_Y
    ) * TERRAIN_LOOKAHEAD_HEIGHT_SCALE_M
    valid = valid_n.reshape(
        env.num_envs, TERRAIN_LOOKAHEAD_SAMPLES_X, TERRAIN_LOOKAHEAD_SAMPLES_Y
    )

    center_y = TERRAIN_LOOKAHEAD_SAMPLES_Y // 2
    y0 = max(0, center_y - 3)
    y1 = min(TERRAIN_LOOKAHEAD_SAMPLES_Y, center_y + 4)
    central_corridor = grid_m[:, :, y0:y1]
    central_valid = valid[:, :, y0:y1]
    # A missing return is not a physical edge. Only compare adjacent cells
    # when both rays returned valid terrain hits; otherwise a zero-filled miss
    # could create a false 14-16 cm wall signature.
    dx_valid = central_valid[:, 1:, :] & central_valid[:, :-1, :]
    dy_valid = central_valid[:, :, 1:] & central_valid[:, :, :-1]
    dx_edges = torch.where(
        dx_valid, torch.abs(torch.diff(central_corridor, dim=1)),
        torch.zeros_like(central_corridor[:, 1:, :]),
    )
    dy_edges = torch.where(
        dy_valid, torch.abs(torch.diff(central_corridor, dim=2)),
        torch.zeros_like(central_corridor[:, :, 1:]),
    )
    edge_peak = torch.maximum(
        dx_edges.amax(dim=(1, 2)),
        dy_edges.amax(dim=(1, 2)),
    )
    valid_fraction = central_valid.float().mean(dim=(1, 2))
    barrier = (edge_peak >= TERRAIN_OBSTACLE_DETOUR_THRESHOLD_M) & (valid_fraction >= 0.50)

    # Occupancy is based on raised surfaces, not small 2-8 cm step tops.
    # The positive/negative halves include the complete expected bypass gates.
    occupied = (grid_m >= TERRAIN_OBSTACLE_DETOUR_THRESHOLD_M) & valid
    neg_occ = occupied[:, :, :center_y].float().sum(dim=(1, 2))
    neg_count = valid[:, :, :center_y].float().sum(dim=(1, 2)).clamp_min(1.0)
    pos_occ = occupied[:, :, center_y + 1 :].float().sum(dim=(1, 2))
    pos_count = valid[:, :, center_y + 1 :].float().sum(dim=(1, 2)).clamp_min(1.0)
    neg_fraction = neg_occ / neg_count
    pos_fraction = pos_occ / pos_count

    difference = torch.abs(neg_fraction - pos_fraction)
    direction = torch.where(pos_fraction <= neg_fraction, 1.0, -1.0)
    direction = torch.where(difference >= 0.015, direction, torch.zeros_like(direction))
    confidence = torch.clamp(difference / 0.20, min=0.0, max=1.0)

    env._robotdog_terrain_obstacle_guidance_cache = {
        "step_key": step_key,
        "barrier": barrier,
        "direction": direction,
        "confidence": confidence,
    }
    return barrier, direction, confidence


def terrain_obstacle_ahead_mask(env, threshold_m: float = TERRAIN_OBSTACLE_DETOUR_THRESHOLD_M):
    """Detect a sharp obstacle edge in the centre of the forward preview."""
    if abs(float(threshold_m) - TERRAIN_OBSTACLE_DETOUR_THRESHOLD_M) > 1.0e-9:
        # The detour threshold is shared with the side-guidance occupancy model.
        raise ValueError("Use TERRAIN_OBSTACLE_DETOUR_THRESHOLD_M consistently for obstacle gating.")
    return _terrain_obstacle_guidance(env)[0]


def obstacle_bypass_alignment_reward(env, command_name: str = "base_velocity", target_lateral_speed: float = 0.12):
    """Reward lateral motion toward the less-obstructed side when a tall wall is ahead.

    This is only active for an approximately straight physical-forward command.
    It is an auxiliary local bypass cue, not a global path planner.
    """
    if target_lateral_speed <= 0.0:
        raise ValueError("target_lateral_speed must be positive")
    command = env.command_manager.get_command(command_name)
    barrier, preferred_side, confidence = _terrain_obstacle_guidance(env)
    forward_command = (command[:, 0] < -0.02) & (torch.abs(command[:, 1]) < 0.04)
    eligible = barrier & forward_command & (preferred_side != 0.0)
    actual_lateral_speed = velocity_xy_in_yaw_frame(env)[:, 1]
    aligned_side_speed = actual_lateral_speed * preferred_side
    reward = torch.clamp(aligned_side_speed / target_lateral_speed, min=0.0, max=1.0)
    return reward * confidence * eligible.float()


class RobotDogAdaptiveVelocityCommand(UniformVelocityCommand):
    """LP-ACRL speed-bin command with omnidirectional XY/yaw modes and ACS ramps."""

    def __init__(self, cfg, env):
        self._robotdog_env = env
        super().__init__(cfg, env)
        self._ensure_acs_buffers()

    def _ensure_acs_buffers(self):
        env = self._robotdog_env
        if not hasattr(self, "_acs_target_speed"):
            self._acs_target_speed = torch.zeros(env.num_envs, device=env.device)
            self._acs_command_speed = torch.zeros(env.num_envs, device=env.device)
            self._acs_target_direction_xy = torch.zeros((env.num_envs, 2), device=env.device)
            self._acs_target_yaw = torch.zeros(env.num_envs, device=env.device)
            self._acs_command_yaw = torch.zeros(env.num_envs, device=env.device)
            self._acs_last_measured_speed = torch.zeros(env.num_envs, device=env.device)

    def _resample_command(self, env_ids: Sequence[int]):
        self._ensure_acs_buffers()
        super()._resample_command(env_ids)

        env = self._robotdog_env
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        if ids.numel() == 0:
            return
        state = _initialize_terrain_curriculum_state(env)
        speed_bins = state["env_speed_bin"][ids].to(dtype=torch.long)
        bounds = torch.as_tensor(SPEED_BIN_BOUNDS_MPS, dtype=torch.float32, device=env.device)
        low = bounds[speed_bins, 0]
        high = bounds[speed_bins, 1]

        sampled_speed = low + torch.rand_like(low) * (high - low)
        fixed_mask = state["env_speed_is_fixed"][ids]
        sampled_speed = torch.where(fixed_mask, state["env_speed_target_mps"][ids], sampled_speed)
        sampled_speed = torch.minimum(torch.maximum(sampled_speed, low), high)

        # Motion modes are balanced enough to teach forward, reverse, lateral,
        # diagonal, and pivoting skills without abandoning the terrain-speed curriculum.
        mode_probabilities = torch.as_tensor(
            MOTION_MODE_PROBABILITIES, dtype=torch.float32, device=env.device
        )
        mode = torch.multinomial(
            mode_probabilities.unsqueeze(0).expand(ids.numel(), -1),
            num_samples=1,
        ).squeeze(-1)
        mode = torch.where(fixed_mask, state["env_motion_mode"][ids].to(dtype=torch.long), mode)

        directions = torch.as_tensor(
            MOTION_MODE_DIRECTIONS_XY, dtype=torch.float32, device=env.device
        )
        target_direction = directions[mode]
        pivot_mask = mode == MOTION_MODE_PIVOT_INDEX
        target_speed = torch.where(pivot_mask, torch.zeros_like(sampled_speed), sampled_speed)

        # Translation episodes occasionally include a signed yaw-rate target,
        # so the policy learns curved paths as well as pure translational modes.
        yaw_active = torch.rand_like(sampled_speed) < YAW_COMMAND_ACTIVE_PROB
        yaw_sign = torch.where(torch.rand_like(sampled_speed) < 0.5, -1.0, 1.0)
        yaw_magnitude = YAW_COMMAND_MIN_RAD_S + torch.rand_like(sampled_speed) * (
            YAW_COMMAND_MAX_RAD_S - YAW_COMMAND_MIN_RAD_S
        )
        target_yaw = torch.where(yaw_active, yaw_sign * yaw_magnitude, torch.zeros_like(sampled_speed))

        # Pivot tasks rotate in place and receive a stronger signed yaw target.
        pivot_yaw_magnitude = PIVOT_YAW_MIN_RAD_S + torch.rand_like(sampled_speed) * (
            PIVOT_YAW_MAX_RAD_S - PIVOT_YAW_MIN_RAD_S
        )
        pivot_yaw_sign = torch.where(torch.rand_like(sampled_speed) < 0.5, -1.0, 1.0)
        target_yaw = torch.where(pivot_mask, pivot_yaw_sign * pivot_yaw_magnitude, target_yaw)

        # Fixed evaluation tasks are deterministic, including lateral and pivot
        # commands; this lets the fixed suite measure the new omni capabilities.
        target_yaw = torch.where(fixed_mask, state["env_yaw_target_rad_s"][ids], target_yaw)
        target_speed = torch.where(fixed_mask, state["env_speed_target_mps"][ids], target_speed)

        # Pivot-only episodes are assigned to the lowest speed bin for LP-ACRL
        # bookkeeping; their competence is measured using angular progress.
        state["env_speed_bin"][ids] = torch.where(
            pivot_mask & (~fixed_mask), torch.zeros_like(speed_bins), speed_bins
        )
        state["env_motion_mode"][ids] = mode
        state["env_speed_target_mps"][ids] = target_speed
        state["env_yaw_target_rad_s"][ids] = target_yaw

        self._acs_target_speed[ids] = target_speed
        self._acs_target_direction_xy[ids] = target_direction
        self._acs_target_yaw[ids] = target_yaw

        initial_speed = torch.where(pivot_mask, torch.zeros_like(low), torch.minimum(low, target_speed))
        initial_speed = torch.where(fixed_mask, torch.minimum(low, target_speed), initial_speed)
        self._acs_command_speed[ids] = initial_speed
        self._acs_command_yaw[ids] = 0.0
        self.vel_command_b[ids, :2] = target_direction * initial_speed.unsqueeze(-1)
        self.vel_command_b[ids, 2] = 0.0

    def _update_command(self):
        self._ensure_acs_buffers()
        super()._update_command()

        max_xy_step = float(self.cfg.acs_slew_mps_per_step)
        max_yaw_step = float(self.cfg.acs_yaw_slew_rad_s_per_step)
        if not math.isfinite(max_xy_step) or max_xy_step <= 0.0:
            raise RuntimeError(f"ACS XY slew must be finite and positive; got {max_xy_step!r}.")
        if not math.isfinite(max_yaw_step) or max_yaw_step <= 0.0:
            raise RuntimeError(f"ACS yaw slew must be finite and positive; got {max_yaw_step!r}.")

        max_command_speed = float(SPEED_BIN_BOUNDS_MPS[-1][1])
        target_speed = self._acs_target_speed.clamp(min=0.0, max=max_command_speed)
        speed_delta = torch.clamp(
            target_speed - self._acs_command_speed,
            min=-max_xy_step,
            max=max_xy_step,
        )
        self._acs_command_speed = (self._acs_command_speed + speed_delta).clamp(
            min=0.0, max=max_command_speed
        )
        self._acs_command_speed = torch.minimum(self._acs_command_speed, target_speed)

        yaw_delta = torch.clamp(
            self._acs_target_yaw - self._acs_command_yaw,
            min=-max_yaw_step,
            max=max_yaw_step,
        )
        self._acs_command_yaw = self._acs_command_yaw + yaw_delta

        self.vel_command_b[:, :2] = self._acs_target_direction_xy * self._acs_command_speed.unsqueeze(-1)
        self.vel_command_b[:, 2] = self._acs_command_yaw

        actual_xy = velocity_xy_in_yaw_frame(self._robotdog_env)
        actual_aligned = torch.sum(actual_xy * self._acs_target_direction_xy, dim=-1)
        self._acs_last_measured_speed = actual_aligned

    def _update_metrics(self):
        """Keep parent metric entries disabled; project metrics are logged by rewards."""
        if isinstance(self.metrics, dict):
            self.metrics.clear()


@configclass
class RobotDogAdaptiveVelocityCommandCfg(UniformVelocityCommandCfg):
    """UniformVelocityCommandCfg extended with XY and yaw acceleration limits."""

    class_type: type = RobotDogAdaptiveVelocityCommand
    acs_slew_mps_per_step: float = ACS_COMMAND_SLEW_MPS_PER_STEP
    acs_yaw_slew_rad_s_per_step: float = ACS_YAW_SLEW_RAD_S_PER_STEP


@configclass
class CommandsCfg:
    """Task-conditioned omnidirectional velocity targets, ramped by ACS."""

    base_velocity = RobotDogAdaptiveVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(EPISODE_LENGTH_S + 1.0, EPISODE_LENGTH_S + 1.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        debug_vis=False,
        acs_slew_mps_per_step=ACS_COMMAND_SLEW_MPS_PER_STEP,
        acs_yaw_slew_rad_s_per_step=ACS_YAW_SLEW_RAD_S_PER_STEP,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-SPEED_BIN_BOUNDS_MPS[-1][1], SPEED_BIN_BOUNDS_MPS[-1][1]),
            lin_vel_y=(-SPEED_BIN_BOUNDS_MPS[-1][1], SPEED_BIN_BOUNDS_MPS[-1][1]),
            ang_vel_z=(-PIVOT_YAW_MAX_RAD_S, PIVOT_YAW_MAX_RAD_S),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_position = FilteredJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"revolute_{i}" for i in range(1, 13)],
        preserve_order=True,
        scale={
            "^revolute_1$": math.radians(30.0),
            "^revolute_2$": math.radians(30.0),
            "^revolute_3$": math.radians(35.0),

            "^revolute_4$": math.radians(30.0),
            "^revolute_5$": math.radians(30.0),
            "^revolute_6$": math.radians(35.0),

            "^revolute_7$": math.radians(30.0),
            "^revolute_8$": math.radians(30.0),
            "^revolute_9$": math.radians(35.0),

            "^revolute_10$": math.radians(30.0),
            "^revolute_11$": math.radians(30.0),
            "^revolute_12$": math.radians(35.0),
        },
        use_default_offset=True,
        alpha=1.0,
        max_delta=2.0,
        deadband=0.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=4, flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=4, flatten_history_dim=True)
        imu_ang_vel = ObsTerm(
            func=_imu_ang_vel_realistic,
            params={"asset_cfg": SceneEntityCfg("imu")},
            history_length=4,
            flatten_history_dim=True,
        )
        imu_lin_acc = ObsTerm(
            func=_imu_lin_acc_realistic,
            params={"asset_cfg": SceneEntityCfg("imu")},
            history_length=4,
            flatten_history_dim=True,
        )
        projected_gravity_est = ObsTerm(func=estimated_projected_gravity, history_length=4, flatten_history_dim=True)
        last_action = ObsTerm(func=mdp.last_action, history_length=4, flatten_history_dim=True)
        # No prescribed gait phase is exposed to the policy.

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        imu_ang_vel_oracle = ObsTerm(func=teacher_imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu")})
        imu_lin_acc_oracle = ObsTerm(func=teacher_imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu")})
        projected_gravity_oracle = ObsTerm(func=teacher_projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        # Privileged teacher input: true base-frame linear velocity (m/s).
        # The actor sees this through obs_groups["actor"] = ["policy", "teacher"].
        base_lin_vel = ObsTerm(func=teacher_base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        # # Vedere de ansamblu asupra întregului teren generat.
        # global_terrain_overview = ObsTerm(
        #     func=teacher_global_terrain_map
        # )
        # global_terrain_validity = ObsTerm(
        #     func=teacher_global_terrain_map_validity
        # )
        # global_map_position = ObsTerm(
        #     func=teacher_global_map_position
        # )
        robot_base_height = ObsTerm(func=teacher_base_height, params={"asset_cfg": SceneEntityCfg("robot")})
        foot_contacts = ObsTerm(func=teacher_foot_contacts, history_length=4, flatten_history_dim=True)
        foot_contact_forces = ObsTerm(func=teacher_foot_contact_forces, history_length=4, flatten_history_dim=True)
        body_contact_states = ObsTerm(func=teacher_body_contact_states)
        # Terrain-aware teacher input appended after the legacy teacher terms.
        # The height map is 17x33 at 5 cm spacing (561 cells), with a paired
        # validity mask. Small range noise/dropout mimic real exteroception.
        terrain_height_map = ObsTerm(
            func=terrain_lookahead_noisy_height_scan,
            params={"sensor_cfg": SceneEntityCfg("terrain_lookahead")},
        )
        terrain_map_validity = ObsTerm(
            func=terrain_lookahead_noisy_validity,
            params={"sensor_cfg": SceneEntityCfg("terrain_lookahead")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=teacher_base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_height = ObsTerm(func=teacher_base_height, params={"asset_cfg": SceneEntityCfg("robot")})
        foot_contact_forces = ObsTerm(func=teacher_foot_contact_forces)
        base_contact_force = ObsTerm(func=teacher_base_contact_force)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


# =============================================================================
# FIXED 2 KG MASS + DETERMINISTIC TEACHER EVENTS
# =============================================================================

TARGET_ROBOT_TOTAL_MASS_KG = 2.0
SERVO_MASS_KG = 0.055
NUM_SERVOS = 12
STRUCTURAL_MASS_KG = TARGET_ROBOT_TOTAL_MASS_KG - NUM_SERVOS * SERVO_MASS_KG

# Approximate CAD/STL volumes from the supplied RobotDog URDF package (cm^3).
# Servo mass is allocated to the link on which each actuator is assumed mounted.
LINK_VOLUME_CM3 = {
    "base_link__2__1__body1": 464.73,
    "servo2_1__body1": 36.91,
    "servo2__1__1__body1": 36.91,
    "servo2_mirror__1__body1": 36.91,
    "servo2_mirror___1__1__body1": 36.91,
    "servo_3_1__body1": 74.26,
    "servo_3__1__1__body1": 74.26,
    "servo_3_mirror__1__body1": 74.26,
    "servo_3_mirror___1__1__body1": 74.26,
    "componentfinal_1__body1": 17.07,
    "componentfinal__1__1__body1": 17.07,
    "componentfinal_mirror__1__body1": 17.07,
    "componentfinal_mirror___1__1__body1": 17.07,
}
SERVO_COUNT_BY_LINK = {
    "base_link__2__1__body1": 4,
    "servo2_1__body1": 1,
    "servo2__1__1__body1": 1,
    "servo2_mirror__1__body1": 1,
    "servo2_mirror___1__1__body1": 1,
    "servo_3_1__body1": 1,
    "servo_3__1__1__body1": 1,
    "servo_3_mirror__1__body1": 1,
    "servo_3_mirror___1__1__body1": 1,
    "componentfinal_1__body1": 0,
    "componentfinal__1__1__body1": 0,
    "componentfinal_mirror__1__body1": 0,
    "componentfinal_mirror___1__1__body1": 0,
}


def set_robotdog_mass_distribution(
    env,
    env_ids: Sequence[int] | None = None,
):
    """Set the robot's link masses to a total of exactly 2 kg."""

    robot = env.scene["robot"]
    body_names = list(robot.body_names)

    # Match the configured CAD links to the actual articulation bodies.
    name_to_index: dict[str, int] = {}

    for expected_name in LINK_VOLUME_CM3:
        matches = [
            i
            for i, actual_name in enumerate(body_names)
            if (
                actual_name == expected_name
                or str(actual_name).endswith(expected_name)
            )
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "[Teacher mass] Expected exactly one body matching "
                f"{expected_name!r}; runtime bodies={body_names!r}. "
                "Cannot safely assign the 2 kg mass distribution."
            )

        name_to_index[expected_name] = matches[0]

    if len(set(name_to_index.values())) != len(name_to_index):
        raise RuntimeError(
            "[Teacher mass] Two CAD links matched the same body."
        )

    if len(body_names) != len(LINK_VOLUME_CM3):
        raise RuntimeError(
            f"[Teacher mass] Expected {len(LINK_VOLUME_CM3)} rigid bodies, "
            f"but found {len(body_names)}: {body_names!r}"
        )

    if sum(SERVO_COUNT_BY_LINK.values()) != NUM_SERVOS:
        raise RuntimeError(
            "[Teacher mass] Servo allocation does not add up to 12."
        )

    # Startup normally applies this to every environment.
    if env_ids is None:
        ids_cpu = torch.arange(
            env.scene.num_envs,
            dtype=torch.long,
            device="cpu",
        )
    else:
        ids_cpu = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device="cpu",
        )

    if ids_cpu.numel() == 0:
        return

    # Allocate 1.34 kg of structural mass proportionally to CAD volume
    # and 0.66 kg of servo mass according to the link allocation.
    total_volume = sum(LINK_VOLUME_CM3.values())

    mass_by_name: dict[str, float] = {}

    for body_name, volume in LINK_VOLUME_CM3.items():
        structural_mass = (
            STRUCTURAL_MASS_KG * volume / total_volume
        )

        motor_mass = (
            SERVO_COUNT_BY_LINK[body_name] * SERVO_MASS_KG
        )

        mass_by_name[body_name] = (
            structural_mass + motor_mass
        )

    if abs(
        sum(mass_by_name.values()) - TARGET_ROBOT_TOTAL_MASS_KG
    ) > 1.0e-6:
        raise RuntimeError(
            "[Teacher mass] Computed masses do not add up to 2.0 kg."
        )

    # Read current PhysX mass and inertia arrays.
    default_mass = robot.data.default_mass.detach()
    default_inertia = robot.data.default_inertia.detach()

    view_mass = robot.root_physx_view.get_masses()
    view_inertia = robot.root_physx_view.get_inertias()

    device = view_mass.device
    dtype = view_mass.dtype

    # Build the target mass vector in the actual articulation body order.
    target_mass = torch.empty(
        (len(body_names),),
        device=device,
        dtype=dtype,
    )

    for body_name, body_mass in mass_by_name.items():
        target_mass[name_to_index[body_name]] = body_mass

    default_mass_first = default_mass[0].to(
        device=device,
        dtype=dtype,
    )

    if torch.any(default_mass_first <= 1.0e-8):
        raise RuntimeError(
            "[Teacher mass] USD contains a zero or invalid default mass."
        )

    # Preserve the CAD-derived inertia proportions while scaling inertia
    # in proportion to the changed mass.
    inertia_scale = target_mass / default_mass_first

    mass_values = (
        target_mass.unsqueeze(0)
        .expand(ids_cpu.numel(), -1)
        .clone()
    )

    inertia_base = default_inertia[0].to(
        device=device,
        dtype=view_inertia.dtype,
    )

    inertia_values_one_env = (
        inertia_base
        * inertia_scale.to(view_inertia.dtype).view(-1, 1)
    )

    inertia_values = (
        inertia_values_one_env.unsqueeze(0)
        .expand(ids_cpu.numel(), -1, -1)
        .clone()
    )

    # Apply updated mass and inertia values through PhysX.
    view_mass[ids_cpu] = mass_values
    robot.root_physx_view.set_masses(view_mass, ids_cpu)

    view_inertia[ids_cpu] = inertia_values
    robot.root_physx_view.set_inertias(view_inertia, ids_cpu)

    # Verify the result instead of assuming the assignment succeeded.
    verified = robot.root_physx_view.get_masses()[ids_cpu]
    total_verified = verified.sum(dim=1)

    if torch.any(
        torch.abs(
            total_verified - TARGET_ROBOT_TOTAL_MASS_KG
        ) > 1.0e-3
    ):
        raise RuntimeError(
            "[Teacher mass] PhysX mass verification failed. "
            f"Measured totals: {total_verified.detach().cpu().tolist()}"
        )

    if int(ids_cpu[0].item()) == 0:
        print(
            f"[Teacher mass] Verified total mass: "
            f"{total_verified.mean().item():.4f} kg",
            flush=True,
        )

        for name in LINK_VOLUME_CM3:
            print(
                f"  {name}: {mass_by_name[name]:.5f} kg",
                flush=True,
            )

        print(
            "[Teacher mass] CAD inertia proportions scaled by mass ratio. "
            "Individual servo centres of mass and inertias are not "
            "modelled separately.",
            flush=True,
        )


def _terrain_task_layout(env):
    """Resolve terrain family columns exactly as Isaac Lab 2.3.1 generates them.

    TerrainGenerator._generate_curriculum_terrains maps each column to a
    sub-terrain using ``column / num_cols + 0.001 < cumulative_proportion``.
    We mirror that public configuration rule rather than assuming fixed column
    ranges, so changing proportions or num_cols remains safe.
    """
    terrain_cfg = env.scene.terrain.cfg.terrain_generator
    sub_terrains = terrain_cfg.sub_terrains
    family_names = list(sub_terrains.keys())
    proportions = np.asarray(
        [float(sub_terrains[name].proportion) for name in family_names],
        dtype=np.float64,
    )
    if len(family_names) == 0 or not np.isfinite(proportions).all() or proportions.sum() <= 0.0:
        raise RuntimeError("LP-ACRL requires valid terrain sub-terrain names and proportions.")
    proportions /= proportions.sum()
    cumulative = np.cumsum(proportions)

    num_cols = int(terrain_cfg.num_cols)
    columns_by_family: list[list[int]] = [[] for _ in family_names]
    family_by_column = []
    for column in range(num_cols):
        matches = np.flatnonzero((column / num_cols + 0.001) < cumulative)
        if matches.size == 0:
            family_index = len(family_names) - 1
        else:
            family_index = int(matches[0])
        family_by_column.append(family_index)
        columns_by_family[family_index].append(column)

    empty_families = [family_names[i] for i, columns in enumerate(columns_by_family) if not columns]
    if empty_families:
        raise RuntimeError(
            "LP-ACRL cannot sample terrain families without generated columns: "
            f"{empty_families}. Increase num_cols or adjust sub-terrain proportions."
        )
    return family_names, columns_by_family, family_by_column


def _build_terrain_speed_feasibility_mask(family_names, num_levels, device):
    """Conservative task-conditioned speed limits for each terrain family/row."""
    mask = torch.zeros(
        (len(family_names), num_levels, NUM_SPEED_BINS), dtype=torch.bool, device=device
    )
    for family_idx, family in enumerate(family_names):
        for level in range(num_levels):
            if family == "flat":
                max_bin = 3
            elif family == "waves":
                max_bin = 2 if level < 5 else 1
            elif family == "discrete_obstacles":
                max_bin = 1 if level < 5 else 0
            elif family == "stairs":
                max_bin = 1 if level < 3 else 0
            elif family == "slopes":
                max_bin = 2 if level < 5 else 1
            elif family == "stepping_stones":
                max_bin = 1 if level < 4 else 0
            else:
                max_bin = 0
            mask[family_idx, level, : max_bin + 1] = True
    if "flat" in family_names:
        flat_idx = family_names.index("flat")
        mask[flat_idx, 1:, :] = False
    active_levels = mask.any(dim=-1)
    if not active_levels.any(dim=-1).all():
        raise RuntimeError("Each terrain family must have at least one feasible terrain-level/speed task.")
    if not active_levels[active_levels].all():
        raise RuntimeError("Every active terrain-level pair must have at least one feasible speed bin.")
    return mask



def _initialize_terrain_curriculum_state(env):
    """Allocate per-episode data and LP-ACRL history for (family, level, speed)."""
    if hasattr(env, "_robotdog_terrain_curriculum_state"):
        return env._robotdog_terrain_curriculum_state

    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        raise RuntimeError("LP-ACRL requires generated terrain_origins.")
    if not all(hasattr(terrain, name) for name in ("terrain_levels", "terrain_types", "env_origins")):
        raise RuntimeError("LP-ACRL requires TerrainImporter terrain_levels/types/env_origins.")

    family_names, columns_by_family, family_by_column = _terrain_task_layout(env)
    num_levels, num_cols = terrain.terrain_origins.shape[:2]
    if num_levels != int(terrain.cfg.terrain_generator.num_rows) or num_cols != int(terrain.cfg.terrain_generator.num_cols):
        raise RuntimeError("Terrain origin tensor does not match TerrainGeneratorCfg rows/columns.")

    device = env.device
    dtype = torch.float32
    n_families = len(family_names)
    n_speed_bins = NUM_SPEED_BINS
    if num_levels == len(LP_ACRL_LEVEL_PRIOR):
        level_prior = torch.tensor(LP_ACRL_LEVEL_PRIOR, dtype=dtype, device=device)
    else:
        level_axis = torch.arange(num_levels, dtype=dtype, device=device)
        level_prior = torch.exp(-level_axis / max(1.0, 0.65 * num_levels))
    level_prior = level_prior / level_prior.sum().clamp_min(1.0e-8)

    family_counts = torch.tensor(
        [len(cols) for cols in columns_by_family], dtype=dtype, device=device
    )
    family_prior = family_counts / family_counts.sum().clamp_min(1.0e-8)
    active_task_mask = _build_terrain_speed_feasibility_mask(family_names, num_levels, device)

    # Normalize priors conditionally. This avoids accidentally under-weighting
    # flat terrain when its duplicate difficulty rows are disabled.
    active_level_mask = active_task_mask.any(dim=-1).to(dtype=dtype)
    family_level_prior = level_prior.unsqueeze(0) * active_level_mask
    family_level_prior = family_level_prior / family_level_prior.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    speed_prior = torch.tensor(LP_ACRL_SPEED_PRIOR, dtype=dtype, device=device)
    speed_prior = speed_prior / speed_prior.sum().clamp_min(1.0e-8)
    conditional_speed_prior = active_task_mask.to(dtype=dtype) * speed_prior.view(1, 1, -1)
    conditional_speed_prior = conditional_speed_prior / conditional_speed_prior.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)

    task_prior = (
        family_prior.view(-1, 1, 1)
        * family_level_prior.unsqueeze(-1)
        * conditional_speed_prior
    )
    task_prior = task_prior * active_task_mask.to(dtype=dtype)
    task_prior = task_prior / task_prior.sum().clamp_min(1.0e-8)

    task_shape = (n_families, num_levels, n_speed_bins)
    state = {
        # Per-episode locomotion metrics; commanded_distance is task target,
        # not the instantaneous ACS-issued command.
        "progress_distance": torch.zeros(env.num_envs, device=device),
        "net_progress_distance": torch.zeros(env.num_envs, device=device),
        "commanded_distance": torch.zeros(env.num_envs, device=device),
        "elapsed_time": torch.zeros(env.num_envs, device=device),
        "clearance_reward_sum": torch.zeros(env.num_envs, device=device),
        "slip_penalty_sum": torch.zeros(env.num_envs, device=device),
        "quality_elapsed": torch.zeros(env.num_envs, device=device),
        # LP-ACRL statistics indexed [family, level, speed bin].
        "task_competence_ema": torch.zeros(task_shape, dtype=dtype, device=device),
        "task_learning_progress": torch.zeros(task_shape, dtype=dtype, device=device),
        "task_improvement_ema": torch.zeros(task_shape, dtype=dtype, device=device),
        "task_regression_ema": torch.zeros(task_shape, dtype=dtype, device=device),
        "task_eval_count": torch.zeros(task_shape, dtype=torch.long, device=device),
        "task_pending_score_sum": torch.zeros(task_shape, dtype=dtype, device=device),
        "task_pending_count": torch.zeros(task_shape, dtype=torch.long, device=device),
        "task_prior": task_prior,
        "active_task_mask": active_task_mask,
        "task_sampling_probs": task_prior.clone(),
        "family_by_column": torch.tensor(family_by_column, dtype=torch.long, device=device),
        "columns_by_family": columns_by_family,
        "family_names": family_names,
        "env_speed_bin": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "env_speed_target_mps": torch.zeros(env.num_envs, dtype=dtype, device=device),
        "env_speed_is_fixed": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
        "env_yaw_target_rad_s": torch.zeros(env.num_envs, dtype=dtype, device=device),
        "env_motion_mode": torch.zeros(env.num_envs, dtype=torch.long, device=device),
        "env_eval_task_index": torch.full((env.num_envs,), -1, dtype=torch.long, device=device),
    }
    env._robotdog_terrain_curriculum_state = state
    print(
        "[LP-ACRL] 3-D task space initialized: "
        f"families={family_names}, rows={num_levels}, speed_bins={n_speed_bins}, "
        f"active_tasks={int(active_task_mask.sum().item())}, "
        f"prior_mix={LP_ACRL_PRIOR_MIX:.2f}, temperature={LP_ACRL_SOFTMAX_TEMPERATURE:.3f}",
        flush=True,
    )
    return state


def _reset_terrain_curriculum_progress(env, env_ids: Sequence[int] | None = None):
    """Reset episode-local accumulators without clearing learned LP-ACRL state."""
    state = _initialize_terrain_curriculum_state(env)
    if env_ids is None:
        env_ids = slice(None)
    for key in (
        "progress_distance", "net_progress_distance", "commanded_distance", "elapsed_time",
        "clearance_reward_sum", "slip_penalty_sum", "quality_elapsed",
    ):
        state[key][env_ids] = 0.0


def _accumulate_terrain_curriculum_progress(env, env_ids: Sequence[int] | None = None):
    """Track signed progress along the active XY command, or angular progress for pivots."""
    state = _initialize_terrain_curriculum_state(env)
    command = env.command_manager.get_command("base_velocity")
    command_xy = command[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=-1)
    command_dir = command_xy / command_speed.clamp_min(1.0e-6).unsqueeze(-1)
    actual_xy = velocity_xy_in_yaw_frame(env)
    aligned_linear_speed = torch.sum(actual_xy * command_dir, dim=-1)

    target_linear_speed = state["env_speed_target_mps"].clamp_min(0.0)
    linear_active = target_linear_speed > 0.02
    target_yaw_speed = state["env_yaw_target_rad_s"]
    pivot_active = (~linear_active) & (torch.abs(target_yaw_speed) > 0.04)
    actual_yaw_speed = env.scene["robot"].data.root_ang_vel_b[:, 2]
    aligned_yaw_speed = actual_yaw_speed * torch.sign(target_yaw_speed)

    aligned_rate = torch.where(linear_active, aligned_linear_speed, aligned_yaw_speed)
    target_rate = torch.where(linear_active, target_linear_speed, torch.abs(target_yaw_speed))
    active = linear_active | pivot_active
    positive_delta = torch.relu(aligned_rate) * POLICY_DT * active.float()
    signed_delta = aligned_rate * POLICY_DT * active.float()
    target_delta = target_rate * POLICY_DT * active.float()

    if env_ids is None:
        state["progress_distance"] += positive_delta
        state["net_progress_distance"] += signed_delta
        state["commanded_distance"] += target_delta
        state["elapsed_time"] += POLICY_DT
    else:
        state["progress_distance"][env_ids] += positive_delta[env_ids]
        state["net_progress_distance"][env_ids] += signed_delta[env_ids]
        state["commanded_distance"][env_ids] += target_delta[env_ids]
        state["elapsed_time"][env_ids] += POLICY_DT


# =============================================================================
# EVENTS / DOMAIN RANDOMIZATION
# =============================================================================
# =============================================================================








def _initialize_net_forward_progress_state(env):
    """Allocate per-environment state for signed step-to-step progress."""
    if hasattr(env, "_robotdog_net_forward_progress_state"):
        return env._robotdog_net_forward_progress_state

    state = {
        "previous_position_w": torch.zeros(
            (env.num_envs, 2), device=env.device
        ),
        "initial_heading_w": torch.zeros(
            (env.num_envs, 2), device=env.device
        ),
        "initialized": torch.zeros(
            (env.num_envs,), dtype=torch.bool, device=env.device
        ),
    }
    env._robotdog_net_forward_progress_state = state
    return state


def reset_net_forward_progress_state(
    env,
    env_ids: Sequence[int] | None = None,
):
    """Reset step-to-step displacement state at every episode reset."""
    state = _initialize_net_forward_progress_state(env)
    if env_ids is None:
        env_ids = slice(None)

    state["initialized"][env_ids] = False
    state["previous_position_w"][env_ids] = 0.0
    state["initial_heading_w"][env_ids] = 0.0


@configclass
class EventCfg:
    set_exact_robot_mass = EventTerm(func=set_robotdog_mass_distribution, mode="startup")

    reset_terrain_curriculum_progress = EventTerm(
        func=_reset_terrain_curriculum_progress,
        mode="reset",
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.01, 0.01),
            "velocity_range": (-0.03, 0.03),
        },
    )
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.000, 0.000),
                "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (math.pi - 0.05, math.pi + 0.05),
            },
            "velocity_range": {
                "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01),
                "roll": (-0.03, 0.03), "pitch": (-0.03, 0.03), "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_net_forward_progress = EventTerm(
        func=reset_net_forward_progress_state, mode="reset"
    )


# =============================================================================
# GAIT-FREE TEACHER REWARDS
# =============================================================================

def commanded_progress_reward_l1_5(
    env,
    command_name="base_velocity",
):
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    command_xy = command[:, :2]
    command_speed = torch.linalg.norm(command_xy, dim=-1)

    # Nu recompensam comenzile de miscare aproape zero.
    moving_command = command_speed > 0.05

    command_dir = (
        command_xy
        / command_speed.clamp_min(0.10).unsqueeze(-1)
    )

    velocity_xy = robot.data.root_lin_vel_b[:, :2]

    projected_velocity = torch.sum(
        velocity_xy * command_dir,
        dim=-1,
    )

    # Numai progresul pozitiv, in directia solicitata.
    forward_progress = torch.clamp(
        projected_velocity,
        min=0.0,
    )

    # Nu recompensam depasirea vitezei comandate.
    capped_progress = torch.minimum(
        forward_progress,
        command_speed,
    )

    # Keep this helper aligned with the highest forward-speed target.
    MAX_REWARDED_SPEED = 0.20

    progress_reward = torch.clamp(
        capped_progress / MAX_REWARDED_SPEED,
        0.0,
        1.0,
    )

    return progress_reward * moving_command.float()


def robotdog_motion_progress_gate(
    env,
    command_name: str = "base_velocity",
    command_threshold: float = 0.02,
    full_speed_mps: float = 0.02,
    full_net_progress_fraction: float = 0.02,
):
    """Gate gait shaping on measured progress for translation or pivot tasks."""
    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=-1)
    command_dir = command_xy / command_speed.clamp_min(1.0e-6).unsqueeze(-1)
    actual_xy = velocity_xy_in_yaw_frame(env)
    aligned_speed = torch.sum(actual_xy * command_dir, dim=-1)

    state = _initialize_terrain_curriculum_state(env)
    target_distance = state["commanded_distance"].clamp_min(1.0e-6)
    net_fraction = state["net_progress_distance"] / target_distance
    progress_gate = torch.clamp(
        net_fraction / max(full_net_progress_fraction, 1.0e-6), 0.0, 1.0
    )

    translation_active = state["env_speed_target_mps"] > command_threshold
    translation_speed_gate = torch.clamp(
        aligned_speed / max(full_speed_mps, 1.0e-6), 0.0, 1.0
    )
    translation_gate = translation_speed_gate * progress_gate

    yaw_target = state["env_yaw_target_rad_s"]
    pivot_active = (~translation_active) & (torch.abs(yaw_target) > 0.04)
    actual_yaw = env.scene["robot"].data.root_ang_vel_b[:, 2]
    aligned_yaw = actual_yaw * torch.sign(yaw_target)
    yaw_speed_gate = torch.clamp(aligned_yaw / 0.15, 0.0, 1.0)
    pivot_gate = yaw_speed_gate * progress_gate

    gate = torch.where(translation_active, translation_gate, pivot_gate)
    active = translation_active | pivot_active
    return gate * active.float()


def velocity_tracking_xy_reward(
    env,
    command_name="base_velocity",
    sigma=0.10,
    moving_command_threshold=0.015,
):
    """Baseline-subtracted velocity tracking; stationary command gets exactly zero."""
    command = env.command_manager.get_command(command_name)
    _accumulate_terrain_curriculum_progress(env)
    target_xy = command[:, :2]
    actual_xy = velocity_xy_in_yaw_frame(env)
    command_speed = torch.linalg.norm(target_xy, dim=-1)
    moving_command = command_speed > moving_command_threshold

    # Compact live diagnostics: physical forward speed, issued command speed,
    # and a linear percentage of command tracking. This is deliberately not
    # the reward itself and cannot create positive reward while standing still.
    command_dir = target_xy / command_speed.clamp_min(1.0e-6).unsqueeze(-1)
    aligned_speed = torch.sum(actual_xy * command_dir, dim=-1)
    tracking_percent = torch.clamp(
        1.0 - torch.abs(aligned_speed - command_speed) / command_speed.clamp_min(0.04),
        min=0.0,
        max=1.0,
    ) * moving_command.float()
    extras = getattr(env, "extras", None)
    if isinstance(extras, dict):
        log = extras.setdefault("log", {})
        log["Metrics/locomotion/mean_command_aligned_speed_mps"] = torch.clamp(
            aligned_speed, min=0.0
        ).mean().detach()
        log["Metrics/locomotion/mean_command_speed_mps"] = command_speed.mean().detach()
        log["Metrics/locomotion/velocity_tracking_percent"] = (
            100.0 * tracking_percent.mean()
        ).detach()

    error_x_sq = torch.square(actual_xy[:, 0] - target_xy[:, 0])
    error_y_sq = torch.square(actual_xy[:, 1] - target_xy[:, 1])
    obstacle_ahead = terrain_obstacle_ahead_mask(env)
    forward_command = (target_xy[:, 0] < -0.02) & (torch.abs(target_xy[:, 1]) < 0.04)
    detour_active = obstacle_ahead & forward_command
    # A high wall (at least 13 cm) temporarily relaxes tracking along the blocked axis and
    # allows lateral displacement toward the open side. The auxiliary bypass
    # reward below supplies a directional cue; clear terrain keeps full tracking.
    forward_error_weight = torch.where(
        detour_active, torch.full_like(command_speed, 0.20), torch.ones_like(command_speed)
    )
    lateral_error_weight = torch.where(
        detour_active, torch.full_like(command_speed, 0.08), torch.ones_like(command_speed)
    )
    error_sq = forward_error_weight * error_x_sq + lateral_error_weight * error_y_sq
    gaussian = torch.exp(-error_sq / max(sigma ** 2, 1.0e-8))
    stationary_baseline = torch.exp(-torch.square(command_speed) / max(sigma ** 2, 1.0e-8))
    tracking = (gaussian - stationary_baseline) / (1.0 - stationary_baseline).clamp_min(1.0e-5)
    tracking = torch.clamp(tracking, min=-1.0, max=1.0)
    motion_gate = robotdog_motion_progress_gate(
        env, command_name=command_name, command_threshold=moving_command_threshold
    )
    moving_reward = tracking * motion_gate
    actual_speed_sq = torch.sum(torch.square(actual_xy), dim=-1)
    standing_reward = torch.exp(-actual_speed_sq / max(sigma ** 2, 1.0e-8)) - 1.0
    return torch.where(moving_command, moving_reward, standing_reward)



def net_forward_progress_reward(
    env,
    command_name="base_velocity",
    command_threshold=0.02,
    speed_deadband_mps=0.005,
):
    """Signed net displacement along the commanded XY direction; ignores pivot-only tasks."""
    robot = env.scene["robot"]
    state = _initialize_net_forward_progress_state(env)
    pos_w = robot.data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_speed = torch.linalg.norm(command_xy, dim=-1)
    command_dir_b = command_xy / command_speed.clamp_min(1.0e-6).unsqueeze(-1)

    q = robot.data.root_quat_w
    qw, qx, qy, qz = q.unbind(dim=-1)
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy.square() + qz.square()),
    )
    c = torch.cos(yaw)
    sn = torch.sin(yaw)
    command_dir_w = torch.stack(
        (c * command_dir_b[:, 0] - sn * command_dir_b[:, 1],
         sn * command_dir_b[:, 0] + c * command_dir_b[:, 1]),
        dim=-1,
    )

    was_initialized = state["initialized"].clone()
    delta_position_w = pos_w - state["previous_position_w"]
    signed_distance = torch.sum(delta_position_w * command_dir_w, dim=-1)
    measured_step_speed = signed_distance / max(POLICY_DT, 1.0e-6)
    deadbanded_speed = torch.sign(measured_step_speed) * torch.relu(
        torch.abs(measured_step_speed) - speed_deadband_mps
    )
    denominator = (command_speed - speed_deadband_mps).clamp_min(0.02)
    reward = torch.clamp(deadbanded_speed / denominator, min=-1.5, max=1.5)
    active = command_speed > command_threshold
    reward = torch.where(was_initialized & active, reward, torch.zeros_like(reward))
    state["previous_position_w"][:] = pos_w
    state["initialized"].fill_(True)
    return reward



def motor_limit_penalty(env, torque_util_threshold: float = 0.85):
    """Penalize sustained torque-limit proximity and clipping per servo."""
    actuator = env.scene["robot"].actuators["all_joints"]
    telemetry = getattr(actuator, "_robotdog_motor_telemetry", None)
    if not isinstance(telemetry, dict):
        raise RuntimeError("motor_limit reward requires per-joint actuator telemetry.")

    samples = telemetry["policy_samples"].unsqueeze(-1).clamp_min(1.0)
    torque_util = telemetry["policy_torque_util_sum"] / samples
    clip_deficit = telemetry["policy_clip_deficit_sum"] / samples
    saturation = telemetry["policy_saturation_sum"] / samples
    proximity = torch.relu(torque_util - torque_util_threshold) / max(
        1.0 - torque_util_threshold, 1.0e-6
    )
    per_joint_penalty = torch.clamp(
        proximity + 0.50 * clip_deficit + 0.25 * saturation, 0.0, 2.0
    )
    penalty = 0.70 * per_joint_penalty.mean(dim=-1) + 0.30 * per_joint_penalty.amax(dim=-1)

    extras = getattr(env, "extras", None)
    if isinstance(extras, dict):
        log = extras.setdefault("log", {})
        log["Metrics/motor/max_joint_torque_util"] = torque_util.amax(dim=-1).mean().detach()
        log["Metrics/motor/max_joint_clip_deficit"] = clip_deficit.amax(dim=-1).mean().detach()
        log["Metrics/motor/max_joint_saturation_fraction"] = saturation.amax(dim=-1).mean().detach()

    # Consume policy-interval buffers; episode counters remain intact for LP-ACRL.
    telemetry["policy_torque_util_sum"].zero_()
    telemetry["policy_clip_deficit_sum"].zero_()
    telemetry["policy_saturation_sum"].zero_()
    telemetry["policy_samples"].zero_()
    return penalty


@configclass
class RewardsCfg:
    # Match the velocity-tracking objective used by common quadruped RL baselines.
    # This custom signed reward peaks at the commanded XY velocity, gives zero
    # at standstill for a non-zero command, and penalizes motion in the wrong direction.
    track_velocity_xy = RewTerm(
        func=velocity_tracking_xy_reward,
        weight=1.4,
        params={
            "command_name": "base_velocity",
            "sigma": 0.10,
            "moving_command_threshold": 0.02,
        },
    )
    track_yaw = RewTerm(
        func=velocity_tracking_yaw,
        weight=0.30,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.04,
            "sigma": 0.18,
        },
    )
    # Signed command-aligned displacement discourages in-place rocking and
    # forward/backward reward exploitation while supporting reverse/lateral commands.
    net_forward_progress = RewTerm(
        func=net_forward_progress_reward,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.02,
        },
    )
    # A forward fall must not be an attractive one-step shortcut to progress.
    # The existing base_contact termination remains active as well.
    base_contact = RewTerm(
        func=base_contact_penalty,
        weight=-50.0,
        params={
            "sensor_cfg": SceneEntityCfg("base_contact"),
            "threshold": 5.0,
            "force_scale": 40.0,
        },
    )
    action_rate = RewTerm(func=final_target_action_rate_l2, weight=-0.06)
    # Moderately relax the acceleration cost; retain regularization against
    # high-frequency joint chatter while allowing more decisive swing motion.
    action_acc = RewTerm(func=joint_acc_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    joint_vel = RewTerm(func=joint_velocity_l2, weight=-0.010, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.002, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
    motor_limit = RewTerm(func=motor_limit_penalty, weight=-0.05, params={"torque_util_threshold": 0.85})
    joint_limit = RewTerm(func=joint_limit_proximity, weight=-0.04, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "margin_fraction": 0.10})
    # Softer posture shaping for initial gait discovery; base contact remains
    # a hard termination and the height penalty remains active.
    orientation = RewTerm(func=orientation_penalty, weight=-0.40, params={"safe_angle": math.radians(5.0), "asset_cfg": SceneEntityCfg("robot")})
    base_roll_pitch = RewTerm(func=base_roll_pitch_penalty, weight=-0.50, params={"asset_cfg": SceneEntityCfg("robot")})
    base_angular_velocity = RewTerm(func=base_angular_velocity_penalty, weight=-0.30, params={"asset_cfg": SceneEntityCfg("robot")})
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.5,
    )
    uncommanded_yaw_rate = RewTerm(
        func=uncommanded_yaw_rate_penalty,
        weight=-1.00,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.01,
            "dead_zone": 0.02,
            "normalization_speed": 0.20,
        },
    )
    base_height = RewTerm(
        func=soft_base_height_penalty,
        weight=-0.40,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("base_ground_scan"),
            "target_height": 0.18,
            "height_scale": 0.04,
            "min_valid_rays": 5,
        },
    )
    foot_slip = RewTerm(func=foot_slip_penalty, weight=-1.0, params={"velocity_scale": 0.20})
    servo3_contact = RewTerm(
        func=servo3_contact_penalty,
        weight=-1.6,
        params={
            "force_scale": 10.0,
            "contact_cost": 1.0,
        },
    )
    # A small support preference prevents unsupported body states without
    # dictating which leg must be lifted.
    support = RewTerm(func=foot_contact_support_reward, weight=0.05)

    # Dense, phase-free lift shaping. Reward increases monotonically up to the
    # 45 mm collision-surface clearance objective.
    foot_clearance = RewTerm(
        func=foot_clearance_reward,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "target_height": GAIT_CLEARANCE_M,
            "moving_command_threshold": 0.02,
        },
    )

    # Touchdown event reward accepts short real lift attempts (50 ms), but the
    # target airtime and 45 mm clearance objective remain independent.
    feet_air_time = RewTerm(
        func=feet_air_time_reward,
        weight=1.60,
        params={
            "target_air_time": 0.16,
            "target_clearance": GAIT_CLEARANCE_M,
            "min_air_time": 0.05,
            "command_name": "base_velocity",
            "moving_command_threshold": 0.02,
        },
    )

    # Event-coupled diagonal timing: liftoff and touchdown should occur close
    # together within each diagonal pair, but are never scheduled by a clock.
    diagonal_event_sync = RewTerm(
        func=diagonal_event_sync_reward,
        weight=0.9,
        params={
            "command_name": "base_velocity",
            "event_window_s": DIAGONAL_EVENT_MATCH_WINDOW_S,
            "sigma_s": DIAGONAL_EVENT_MATCH_SIGMA_S,
            "velocity_threshold": 0.02,
        },
    )
    # Weak contact-based preference for the opposite diagonal to maintain
    # support while one diagonal is airborne. It does not set swing duration.
    diagonal_alternation = RewTerm(
        func=diagonal_alternation_reward,
        weight=0.05,
        params={"command_name": "base_velocity", "velocity_threshold": 0.02},
    )
    # Mild forward-speed deficit penalty; clearance and diagonal coupling
    # are driven by measured events rather than a prescribed gait phase.
    obstacle_bypass = RewTerm(
        func=obstacle_bypass_alignment_reward,
        weight=0.45,
        params={"command_name": "base_velocity", "target_lateral_speed": 0.12},
    )
    forward_deficit = RewTerm(
        func=forward_velocity_deficit_penalty,
        weight=-0.35,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.03,
            "speed_tolerance": 0.005,
            "normalization_speed": 0.08,
        },
    )
    foot_impact = RewTerm(
        func=foot_impact_penalty,
        weight=-0.05,
        params={"force_scale": 80.0},
    )

# =============================================================================
# TERMINATIONS
# =============================================================================

@configclass
class TerminationsCfg:

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    base_contact = DoneTerm(
        func=base_contact_termination,
        params={
            "sensor_cfg": SceneEntityCfg("base_contact"),
            "threshold": 5.0,
        },
    )


def _motor_episode_metrics(env, env_ids):
    """Compact whole-robot motor-health statistics for LP-ACRL."""
    actuator = env.scene["robot"].actuators["all_joints"]
    telemetry = getattr(actuator, "_robotdog_motor_telemetry", None)
    if not isinstance(telemetry, dict):
        raise RuntimeError("LP-ACRL requires DelayedDCMotor per-joint telemetry.")
    samples = telemetry["episode_samples"][env_ids].unsqueeze(-1).clamp_min(1.0)
    torque = telemetry["torque_util_sum"][env_ids] / samples
    speed = telemetry["speed_util_sum"][env_ids] / samples
    clipping = telemetry["clip_deficit_sum"][env_ids] / samples
    saturation = telemetry["saturation_sum"][env_ids] / samples
    return {
        "torque_mean": torque.mean(dim=-1), "torque_max": torque.amax(dim=-1),
        "speed_mean": speed.mean(dim=-1), "speed_max": speed.amax(dim=-1),
        "clip_mean": clipping.mean(dim=-1), "clip_max": clipping.amax(dim=-1),
        "saturation_mean": saturation.mean(dim=-1), "saturation_max": saturation.amax(dim=-1),
    }



def _episode_competence_score(env, state, env_ids):
    """Net-progress-based competence; stationary/in-place episodes score zero."""
    elapsed = state["elapsed_time"][env_ids].clamp_min(1.0e-6)
    target_distance = state["commanded_distance"][env_ids].clamp_min(1.0e-6)
    tracking = torch.clamp(state["net_progress_distance"][env_ids] / target_distance, 0.0, 1.0)
    movement_gate = torch.clamp(tracking / 0.10, 0.0, 1.0)
    survival = torch.clamp(elapsed / EPISODE_LENGTH_S, 0.0, 1.0)
    quality_elapsed = state["quality_elapsed"][env_ids].clamp_min(1.0e-6)
    clearance = torch.clamp(state["clearance_reward_sum"][env_ids] / quality_elapsed, 0.0, 1.0)
    mean_slip = torch.clamp(state["slip_penalty_sum"][env_ids] / quality_elapsed, 0.0, 5.0)
    slip_quality = torch.exp(-mean_slip)
    motor = _motor_episode_metrics(env, env_ids)
    torque_excess = torch.clamp((motor["torque_max"] - 0.75) / 0.25, 0.0, 1.0)
    speed_excess = torch.clamp((motor["speed_max"] - 0.90) / 0.10, 0.0, 1.0)
    motor_health = torch.clamp(
        1.0 - 0.35 * motor["saturation_max"]
        - 0.30 * torch.clamp(motor["clip_max"], 0.0, 1.0)
        - 0.20 * torque_excess - 0.15 * speed_excess, 0.0, 1.0
    )
    raw_score = (
        LP_ACRL_SCORE_WEIGHT_TRACKING * tracking
        + LP_ACRL_SCORE_WEIGHT_SURVIVAL * survival
        + LP_ACRL_SCORE_WEIGHT_CLEARANCE * clearance
        + LP_ACRL_SCORE_WEIGHT_SLIP * slip_quality
        + LP_ACRL_SCORE_WEIGHT_MOTOR * motor_health
    )
    score = raw_score * movement_gate
    components = {
        "tracking": tracking, "movement_gate": movement_gate,
        "survival": survival, "clearance": clearance,
        "slip_quality": slip_quality, "motor_health": motor_health,
        "torque_util": motor["torque_mean"], "torque_util_max": motor["torque_max"],
        "speed_util": motor["speed_mean"], "speed_util_max": motor["speed_max"],
        "clip_deficit": motor["clip_mean"], "clip_deficit_max": motor["clip_max"],
        "saturation_fraction": motor["saturation_mean"],
        "saturation_fraction_max": motor["saturation_max"],
    }
    return torch.clamp(score, 0.0, 1.0), components




# (terrain family, difficulty row, speed bin, motion mode, signed yaw target).
# Mode indices follow MOTION_MODE_DIRECTIONS_XY; pivot mode 6 uses zero XY speed.
FIXED_EVAL_TASKS = (
    ("flat", 0, 0, 0, 0.0), ("flat", 0, 1, 0, 0.0),
    ("flat", 0, 2, 0, 0.0), ("flat", 0, 3, 0, 0.0),
    ("waves", 2, 2, 0, 0.0), ("waves", 5, 1, 0, 0.0),
    ("waves", 8, 1, 0, 0.0), ("waves", 9, 0, 0, 0.0),
    ("discrete_obstacles", 2, 1, 0, 0.0),
    ("discrete_obstacles", 5, 0, 0, 0.0),
    ("discrete_obstacles", 9, 0, 0, 0.0),
    ("stairs", 2, 1, 0, 0.0), ("stairs", 5, 0, 0, 0.0),
    ("stairs", 9, 0, 0, 0.0),
    ("slopes", 2, 2, 0, 0.0), ("slopes", 5, 1, 0, 0.0),
    ("slopes", 8, 1, 0, 0.0),
    ("stepping_stones", 2, 1, 0, 0.0), ("stepping_stones", 6, 0, 0, 0.0),
    ("flat", 0, 1, 2, 0.0), ("flat", 0, 1, 3, 0.0),
    ("flat", 0, 0, 6, 0.60), ("flat", 0, 0, 6, -0.60),
)


def fixed_evaluation_task_assignment(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    """Assign a deterministic evaluation suite and collect completed task scores."""
    del asset_cfg
    terrain = env.scene.terrain
    state = _initialize_terrain_curriculum_state(env)
    ids = (
        torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    )
    if ids.numel() == 0:
        return {"fixed_eval_assigned_envs": torch.zeros((), device=env.device)}

    if not hasattr(env, "_robotdog_eval_last_scores"):
        env._robotdog_eval_last_scores = torch.zeros(env.num_envs, device=env.device)
        env._robotdog_eval_last_task_ids = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        env._robotdog_eval_last_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    # At eval start/end, partial training/evaluation episodes are explicitly
    # discarded. Only actual completed episodes enter the fixed evaluation.
    ignore_completion = bool(getattr(env, "_robotdog_eval_ignore_next_completion", False))
    if not ignore_completion:
        completed = env.episode_length_buf[ids] > 0
        if torch.any(completed):
            completed_ids = ids[completed]
            scores, _ = _episode_competence_score(env, state, completed_ids)
            env._robotdog_eval_last_scores[completed_ids] = scores
            env._robotdog_eval_last_task_ids[completed_ids] = state["env_eval_task_index"][completed_ids]
            env._robotdog_eval_last_valid[completed_ids] = True
    else:
        env._robotdog_eval_last_valid[ids] = False

    task_count = len(FIXED_EVAL_TASKS)
    env._robotdog_eval_task_count = task_count
    task_ids = ids.remainder(task_count)
    levels = torch.empty_like(ids)
    speed_bins = torch.empty_like(ids)
    columns = torch.empty_like(ids)
    fixed_modes = torch.empty_like(ids)
    fixed_yaws = torch.empty(ids.numel(), dtype=torch.float32, device=env.device)
    fixed_speeds = torch.empty(ids.numel(), dtype=torch.float32, device=env.device)
    speed_bounds = torch.as_tensor(SPEED_BIN_BOUNDS_MPS, dtype=torch.float32, device=env.device)
    family_to_index = {name: idx for idx, name in enumerate(state["family_names"])}

    for task_index, (family_name, level, speed_bin, motion_mode, yaw_target) in enumerate(FIXED_EVAL_TASKS):
        if family_name not in family_to_index:
            raise RuntimeError(f"Fixed eval terrain family '{family_name}' is not generated.")
        family_idx = family_to_index[family_name]
        if (
            level >= state["active_task_mask"].shape[1]
            or speed_bin >= NUM_SPEED_BINS
            or motion_mode < 0
            or motion_mode >= len(MOTION_MODE_DIRECTIONS_XY)
        ):
            raise RuntimeError(f"Fixed evaluation task {(family_name, level, speed_bin, motion_mode, yaw_target)} is out of range.")
        if not bool(state["active_task_mask"][family_idx, level, speed_bin].item()):
            raise RuntimeError(f"Fixed evaluation task {(family_name, level, speed_bin)} is not feasible.")
        select = task_ids == task_index
        if not torch.any(select):
            continue
        levels[select] = level
        speed_bins[select] = speed_bin
        fixed_modes[select] = motion_mode
        family_columns = state["columns_by_family"][family_idx]
        available_columns = torch.as_tensor(family_columns, dtype=torch.long, device=env.device)
        column_offset = (ids[select] // task_count).remainder(available_columns.numel())
        columns[select] = available_columns[column_offset]
        fixed_speeds[select] = (
            0.0 if motion_mode == MOTION_MODE_PIVOT_INDEX
            else 0.5 * (speed_bounds[speed_bin, 0] + speed_bounds[speed_bin, 1])
        )
        fixed_yaws[select] = float(yaw_target)

    terrain.terrain_levels[ids] = levels
    terrain.terrain_types[ids] = columns
    terrain.env_origins[ids] = terrain.terrain_origins[levels, columns]
    state["env_speed_bin"][ids] = speed_bins
    state["env_speed_target_mps"][ids] = fixed_speeds
    state["env_speed_is_fixed"][ids] = True
    state["env_yaw_target_rad_s"][ids] = fixed_yaws
    state["env_motion_mode"][ids] = fixed_modes
    state["env_eval_task_index"][ids] = task_ids
    return {
        "fixed_eval_assigned_envs": torch.tensor(float(ids.numel()), device=env.device),
        "fixed_eval_mean_target_speed_mps": fixed_speeds.mean(),
    }


def terrain_levels_progressive(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    """LP-ACRL over feasible (terrain family, terrain level, forward-speed bin)."""
    if bool(getattr(env, "_robotdog_fixed_eval_mode", False)):
        return fixed_evaluation_task_assignment(env, env_ids, asset_cfg)

    del asset_cfg
    terrain = env.scene.terrain
    state = _initialize_terrain_curriculum_state(env)
    device = env.device
    ids = (
        torch.arange(env.num_envs, dtype=torch.long, device=device)
        if env_ids is None
        else torch.as_tensor(env_ids, dtype=torch.long, device=device)
    )
    if ids.numel() == 0:
        return {"mean_level": terrain.terrain_levels.float().mean()}

    n_families, n_levels, n_speed_bins = state["task_competence_ema"].shape
    family_by_column = state["family_by_column"]
    completed_levels = terrain.terrain_levels[ids].to(dtype=torch.long).clone()
    completed_columns = terrain.terrain_types[ids].to(dtype=torch.long).clone()
    completed_families = family_by_column[completed_columns]
    if "flat" in state["family_names"]:
        flat_idx = state["family_names"].index("flat")
        completed_levels = torch.where(
            completed_families == flat_idx, torch.zeros_like(completed_levels), completed_levels
        )
    completed_speed_bins = state["env_speed_bin"][ids].clamp(0, n_speed_bins - 1)
    flat_task_ids = (
        (completed_families * n_levels + completed_levels) * n_speed_bins
        + completed_speed_bins
    )

    completed_mask = env.episode_length_buf[ids] > 0
    zero = torch.zeros((), dtype=torch.float32, device=device)
    positive_speed = net_speed = zero.view(1)
    mean_competence = mean_tracking = mean_survival = mean_clearance = zero
    # No episode may have completed on the environment's first reset. Keep
    # curriculum diagnostics defined in that case; `components` is only
    # created after at least one completed episode has been scored.
    mean_slip_quality = mean_motor_health = zero

    if torch.any(completed_mask):
        valid_ids = ids[completed_mask]
        valid_task_ids = flat_task_ids[completed_mask]
        scores, components = _episode_competence_score(env, state, valid_ids)
        num_tasks = n_families * n_levels * n_speed_bins
        batch_sum = torch.bincount(valid_task_ids, weights=scores, minlength=num_tasks).reshape(
            n_families, n_levels, n_speed_bins
        )
        batch_count = torch.bincount(valid_task_ids, minlength=num_tasks).reshape(
            n_families, n_levels, n_speed_bins
        )
        state["task_pending_score_sum"] += batch_sum
        state["task_pending_count"] += batch_count
        ready = state["task_pending_count"] >= LP_ACRL_MIN_EPISODES_PER_TASK

        if torch.any(ready):
            batch_mean = state["task_pending_score_sum"] / state["task_pending_count"].clamp_min(1)
            old_ema = state["task_competence_ema"].clone()
            first_eval = state["task_eval_count"] == 0
            new_ema = (
                (1.0 - LP_ACRL_SCORE_EMA_ALPHA) * old_ema
                + LP_ACRL_SCORE_EMA_ALPHA * batch_mean
            )
            new_ema = torch.where(first_eval, batch_mean, new_ema)
            delta = new_ema - old_ema
            delta = torch.where(first_eval, torch.zeros_like(delta), delta)
            magnitude = torch.abs(delta)
            improvement = torch.relu(delta)
            regression = torch.relu(-delta)

            new_lp = (
                (1.0 - LP_ACRL_PROGRESS_EMA_ALPHA) * state["task_learning_progress"]
                + LP_ACRL_PROGRESS_EMA_ALPHA * magnitude
            )
            new_improvement = (
                (1.0 - LP_ACRL_PROGRESS_EMA_ALPHA) * state["task_improvement_ema"]
                + LP_ACRL_PROGRESS_EMA_ALPHA * improvement
            )
            new_regression = (
                (1.0 - LP_ACRL_PROGRESS_EMA_ALPHA) * state["task_regression_ema"]
                + LP_ACRL_PROGRESS_EMA_ALPHA * regression
            )
            new_lp = torch.where(first_eval, torch.zeros_like(new_lp), new_lp)
            new_improvement = torch.where(first_eval, torch.zeros_like(new_improvement), new_improvement)
            new_regression = torch.where(first_eval, torch.zeros_like(new_regression), new_regression)

            state["task_competence_ema"] = torch.where(ready, new_ema, old_ema)
            state["task_learning_progress"] = torch.where(
                ready, new_lp, state["task_learning_progress"]
            )
            state["task_improvement_ema"] = torch.where(
                ready, new_improvement, state["task_improvement_ema"]
            )
            state["task_regression_ema"] = torch.where(
                ready, new_regression, state["task_regression_ema"]
            )
            state["task_eval_count"] += ready.to(dtype=torch.long)
            state["task_pending_score_sum"] = torch.where(
                ready, torch.zeros_like(state["task_pending_score_sum"]), state["task_pending_score_sum"]
            )
            state["task_pending_count"] = torch.where(
                ready, torch.zeros_like(state["task_pending_count"]), state["task_pending_count"]
            )

            # Absolute competence change prioritizes both newly learned tasks
            # and tasks whose competence is moving in the wrong direction.
            lp = torch.clamp(state["task_learning_progress"], min=0.0, max=LP_ACRL_LP_CLIP)
            logits = lp / LP_ACRL_SOFTMAX_TEMPERATURE + torch.log(
                state["task_prior"].clamp_min(1.0e-8)
            )
            logits = torch.where(
                state["active_task_mask"], logits, torch.full_like(logits, -1.0e9)
            )
            lp_distribution = torch.softmax(logits.reshape(-1), dim=0).reshape(
                n_families, n_levels, n_speed_bins
            )
            lp_distribution = lp_distribution * state["active_task_mask"].to(dtype=lp_distribution.dtype)
            lp_distribution = lp_distribution / lp_distribution.sum().clamp_min(1.0e-8)
            mixed = (
                (1.0 - LP_ACRL_PRIOR_MIX) * lp_distribution
                + LP_ACRL_PRIOR_MIX * state["task_prior"]
            )
            mixed = mixed * state["active_task_mask"].to(dtype=mixed.dtype)
            state["task_sampling_probs"] = mixed / mixed.sum().clamp_min(1.0e-8)

        elapsed = state["elapsed_time"][valid_ids].clamp_min(1.0e-6)
        translation_mask = state["env_speed_target_mps"][valid_ids] > 0.02
        positive_speed = (state["progress_distance"][valid_ids] / elapsed) * translation_mask.float()
        net_speed = (state["net_progress_distance"][valid_ids] / elapsed) * translation_mask.float()
        mean_competence = scores.mean()
        mean_tracking = components["tracking"].mean()
        mean_survival = components["survival"].mean()
        mean_clearance = components["clearance"].mean()
        mean_slip_quality = components["slip_quality"].mean()
        mean_motor_health = components["motor_health"].mean()

    # Sample family, row and speed bin jointly from LP-ACRL's 3-D distribution.
    flat_probs = state["task_sampling_probs"].reshape(-1)
    sampled = torch.multinomial(flat_probs, ids.numel(), replacement=True)
    sampled_family = torch.div(sampled, n_levels * n_speed_bins, rounding_mode="floor")
    remainder = sampled.remainder(n_levels * n_speed_bins)
    sampled_level = torch.div(remainder, n_speed_bins, rounding_mode="floor")
    sampled_speed_bin = remainder.remainder(n_speed_bins)
    sampled_column = torch.empty_like(sampled_family)
    for family_idx, family_columns in enumerate(state["columns_by_family"]):
        family_mask = sampled_family == family_idx
        count = int(family_mask.sum().item())
        if count:
            columns = torch.as_tensor(family_columns, dtype=torch.long, device=device)
            selection = torch.randint(0, columns.numel(), (count,), device=device)
            sampled_column[family_mask] = columns[selection]

    terrain.terrain_levels[ids] = sampled_level
    terrain.terrain_types[ids] = sampled_column
    terrain.env_origins[ids] = terrain.terrain_origins[sampled_level, sampled_column]
    state["env_speed_bin"][ids] = sampled_speed_bin
    state["env_speed_target_mps"][ids] = 0.0  # populated by ACS resampling below
    state["env_speed_is_fixed"][ids] = False
    state["env_yaw_target_rad_s"][ids] = 0.0
    state["env_motion_mode"][ids] = 0
    state["env_eval_task_index"][ids] = -1

    probs = state["task_sampling_probs"]
    active = state["active_task_mask"]
    entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum()
    output = {
        # Terrain progression and actual episode-level locomotion.
        "mean_level": terrain.terrain_levels.float().mean(),
        "mean_command_aligned_progress_speed_mps": positive_speed.mean(),
        "mean_net_command_aligned_speed_mps": net_speed.mean(),
        "mean_competence_score": mean_competence,
        "mean_tracking_fraction": mean_tracking,
        "mean_survived_fraction": mean_survival,
        # Traversal quality and sim-to-real actuator health.
        "mean_clearance_competence": mean_clearance,
        "mean_slip_quality": mean_slip_quality,
        "mean_motor_health": mean_motor_health,
        # LP-ACRL sampler coverage/progress diagnostics.
        "lp_task_entropy": entropy,
        "lp_evaluated_task_fraction": (
            ((state["task_eval_count"] > 0) & active).sum().float() / active.sum().clamp_min(1)
        ),
    }
    return output


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=terrain_levels_progressive)

# =============================================================================
# ENV CONFIG
# =============================================================================

@configclass
class RobotdogTeacherStudentVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """LP-ACRL teacher with noisy local elevation scan and omni-directional motion modes."""

    scene: RobotdogSceneCfg = RobotdogSceneCfg(num_envs=1024, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        forward_range = self.commands.base_velocity.ranges.lin_vel_x
        yaw_range = self.commands.base_velocity.ranges.ang_vel_z
        print("========== ROBOTDOG TEACHER V2: TERRAIN-SCAN OMNIDIRECTIONAL LOCOMOTION ==========")
        print("USD:", self.scene.robot.spawn.usd_path)
        print("num_envs:", self.scene.num_envs)
        print("Curriculum: synthetic terrain-command + competence-based 3-D LP-ACRL")
        lateral_range = self.commands.base_velocity.ranges.lin_vel_y
        print(
            "Commands: LP-ACRL translation speed bins "
            f"{SPEED_BIN_BOUNDS_MPS}; modes=forward/reverse/left/right/diagonals/pivot; "
            f"lateral envelope={lateral_range[0]:+.2f}..{lateral_range[1]:+.2f} m/s; "
            f"yaw envelope={yaw_range[0]:+.2f}..{yaw_range[1]:+.2f} rad/s; "
            "episode targets are ACS-ramped at 50 Hz"
        )
        print(
            "LP-ACRL task sampler: terrain family x difficulty row x speed bin; "
            f"{LP_ACRL_MIN_EPISODES_PER_TASK} episodes/task evaluation batch; "
            f"softmax temperature={LP_ACRL_SOFTMAX_TEMPERATURE:.3f}; "
            f"prior mix={LP_ACRL_PRIOR_MIX:.2f}"
        )
        print(
            "Terrain preview: actor-visible noisy elevation scan "
            f"{TERRAIN_LOOKAHEAD_SIZE_M[0]:.2f}x{TERRAIN_LOOKAHEAD_SIZE_M[1]:.2f} m, "
            f"{TERRAIN_LOOKAHEAD_RESOLUTION_M:.2f} m resolution, "
            f"{TERRAIN_LOOKAHEAD_RAY_COUNT} rays, "
            f"height scale={TERRAIN_LOOKAHEAD_HEIGHT_SCALE_M:.2f} m, "
            f"noise={TERRAIN_LOOKAHEAD_NOISE_STD_M * 1000.0:.1f} mm"
        )
        # Track commanded yaw explicitly while suppressing only uncommanded yaw.
        yaw_tracking_term = getattr(self.rewards, "track_yaw", None)
        yaw_tracking_weight = float(yaw_tracking_term.weight) if yaw_tracking_term is not None else 0.0
        uncommanded_yaw_weight = float(self.rewards.uncommanded_yaw_rate.weight)
        print(
            "Active locomotion reward: Gaussian XY tracking "
            f"({self.rewards.track_velocity_xy.weight:.2f}); signed progress "
            f"({self.rewards.net_forward_progress.weight:.2f}); yaw tracking "
            f"{'enabled' if yaw_tracking_weight != 0.0 else 'disabled'} "
            f"(weight={yaw_tracking_weight:.2f}); uncommanded-yaw penalty="
            f"{uncommanded_yaw_weight:.2f}"
        )
        print("Active locomotion structure: event-driven contacts + learned XY/yaw commands; no prescribed gait clock")
        print("No global CPG phase observation, fixed gait frequency, or phase-conditioned swing reward is registered")
        print("Motion modes: forward/reverse/left/right/diagonals/pivot; high-wall bypass guidance enabled")
        print("Barrier task: staggered 14-16 cm walls with alternating 50-60 cm lateral gates")
        print(f"Fixed evaluation suite includes {len(FIXED_EVAL_TASKS)} terrain and omnidirectional/yaw tasks")
        print(
            "Clearance: actual USD collision support points vs nearest terrain ray; "
            f"target={GAIT_CLEARANCE_M * 100.0:.1f} cm; ray grid={FOOT_CLEARANCE_RAY_GRID_SIZE_M} m "
            f"at {FOOT_CLEARANCE_RAY_RESOLUTION_M * 100.0:.1f} cm spacing, centered on validated sole offsets"
        )
        print(
            "Touchdown airtime shaping: minimum event interval 0.05 s; "
            f"target airtime 0.16 s; target clearance {GAIT_CLEARANCE_M * 100.0:.1f} cm"
        )
        print(
            "Foot event contact: terrain-filtered force_matrix_w with hysteresis, "
            f"ON={FOOT_CONTACT_ON_THRESHOLD_N:.2f} N / OFF={FOOT_CONTACT_OFF_THRESHOLD_N:.2f} N; "
            f"foot history={DECIMATION} physics samples"
        )
        print("Teacher actor receives proprioception, privileged state, omnidirectional commands, and noisy local elevation/validity")
        print("Obstacle bypass cue is local and scan-conditioned; no global route planner is included")
        print("Actions: PPO controls all 12 filtered joint targets; no IK or joint trajectory generator")
        print(f"Servo command deadband: {SERVO_COMMAND_DEADBAND_DEG:.2f} deg; physical position-error deadzone: {POSITION_ERROR_DEADBAND_DEG:.2f} deg")
        print("Servo3 terrain-contact penalty is averaged across four legs before applying its reward weight")
        print(
            "STS3215 torque-speed model + motor telemetry; actuator variation: "
            f"torque {SERVO_TORQUE_SCALE_MIN:.2f}..{SERVO_TORQUE_SCALE_MAX:.2f}, "
            f"speed {SERVO_SPEED_SCALE_MIN:.2f}..{SERVO_SPEED_SCALE_MAX:.2f}, "
            f"delay {ACTUATOR_DELAY_MIN}..{ACTUATOR_DELAY_MAX} physics steps"
        )
        print("Teacher uses a 561-ray local elevation scan plus privileged state; student distillation is a later stage")
        print("Terrain curriculum: competence-based LP-ACRL over feasible terrain-speed tasks; 20% prior floor")
        print("===============================================================", flush=True)

        self.decimation = DECIMATION
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = PHYSICS_DT
        self.sim.render_interval = DECIMATION
        print(
            f"Synthetic terrain set: {len(self.scene.terrain.terrain_generator.sub_terrains)} families; "
            f"grid={self.scene.terrain.terrain_generator.horizontal_scale:.3f} m; "
            f"speed bins={SPEED_BIN_BOUNDS_MPS}; ACS ramp={ACS_COMMAND_SLEW_MPS_PER_STEP:.3f} m/s/policy-step (bin floor to target)",
            flush=True,
        )
        print("LP-ACRL scores net tracking, survival, clearance, slip and motor health; stationary episodes score zero")
        print(
            f"Sim2Real envelope: body={ROBOT_BODY_LENGTH_M:.2f} m; leg={ROBOT_LEG_MAX_EXTENSION_M:.2f} m; "
            f"STS3215={STS3215_SUPPLY_V:.1f} V, rated={STS3215_RATED_TORQUE_NM:.3f} Nm, "
            f"stall={STS3215_PEAK_TORQUE_NM:.3f} Nm, no-load={math.degrees(STS3215_NO_LOAD_SPEED):.0f} deg/s, "
            f"rated/stall current={STS3215_RATED_CURRENT_A:.2f}/{STS3215_STALL_CURRENT_A:.2f} A; "
            f"max task speed={SPEED_BIN_BOUNDS_MPS[-1][1]:.2f} m/s",
            flush=True,
        )
        print("PPO self-correction is trainer-side; use run_training_with_self_correction()")
        # Use the terrain's material for the simulation ground collider.
        self.sim.physics_material = self.scene.terrain.physics_material
        # TerrainGenerator.curriculum=True is required to pre-generate ordered
        # difficulty rows. LP-ACRL then samples arbitrary row/column origins.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


        # Keep the solver and GPU buffers used by the original RobotDog environment.
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.max_position_iteration_count = 8
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.max_velocity_iteration_count = 2
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**22
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22
        self.sim.physx.gpu_collision_stack_size = 2**27
        self.sim.physx.gpu_temp_buffer_capacity = 2**25

        # Keep STS3215 nominal ratings and PD gains fixed; randomize only small
        # per-environment/per-joint variation around those nominal values.
        actuator = self.scene.robot.actuators["all_joints"]
        actuator.stiffness = SERVO_KP
        actuator.damping = SERVO_KD
        actuator.saturation_effort = STS3215_PEAK_TORQUE_NM
        actuator.effort_limit = STS3215_RATED_TORQUE_NM
        actuator.effort_limit_sim = STS3215_PEAK_TORQUE_NM
        actuator.velocity_limit = STS3215_NO_LOAD_SPEED
        actuator.velocity_limit_sim = STS3215_NO_LOAD_SPEED
        actuator.target_velocity_limit = JOINT_TARGET_MAX_VELOCITY_RAD_S
        actuator.target_acceleration_limit = JOINT_TARGET_MAX_ACCELERATION_RAD_S2
        actuator.min_delay = ACTUATOR_DELAY_MIN
        actuator.max_delay = ACTUATOR_DELAY_MAX
        actuator.backlash_min = BACKLASH_MIN_RAD
        actuator.backlash_max = BACKLASH_MAX_RAD
        actuator.torque_scale_min = SERVO_TORQUE_SCALE_MIN
        actuator.torque_scale_max = SERVO_TORQUE_SCALE_MAX
        actuator.speed_scale_min = SERVO_SPEED_SCALE_MIN
        actuator.speed_scale_max = SERVO_SPEED_SCALE_MAX
        actuator.zero_offset_min_rad = SERVO_ZERO_OFFSET_MIN_RAD
        actuator.zero_offset_max_rad = SERVO_ZERO_OFFSET_MAX_RAD


# Kept as a naming alias for task registration / later distillation wiring; no
# additional user-defined config inheritance is used.
RobotdogStudentEnvCfg = RobotdogTeacherStudentVelocityEnvCfg



ROBOTDOG_ARCHITECTURE_VERSION = "teacher-v2-terrain-scan-omnidirectional-staggered-barriers-v2"
ROBOTDOG_PPO_SELF_CORRECTION_SETTINGS = {
    "enabled": PPO_SELF_CORRECTION_ENABLED,
    "evaluation_interval": PPO_SELF_CORRECTION_EVAL_INTERVAL,
    "patience": PPO_SELF_CORRECTION_PATIENCE,
    "improvement_epsilon": PPO_SELF_CORRECTION_IMPROVEMENT_EPS,
    "regression_tolerance": PPO_SELF_CORRECTION_REGRESSION_TOL,
    "learning_rate_multiplier": PPO_SELF_CORRECTION_LR_MULTIPLIER,
    "minimum_learning_rate": PPO_SELF_CORRECTION_MIN_LR,
    "action_std_multiplier": PPO_SELF_CORRECTION_STD_MULTIPLIER,
    "minimum_action_std": PPO_SELF_CORRECTION_MIN_STD,
    "maximum_action_std": PPO_SELF_CORRECTION_MAX_STD,
}
