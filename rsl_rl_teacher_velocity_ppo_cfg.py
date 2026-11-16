from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RobotDogTeacherVelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Teacher V2 PPO for local elevation scans and omnidirectional locomotion.

    Legacy actor/critic weights are expanded by train_robotdog_teacher_v2.py;
    the 1122 new scan features start with zero first-layer weights.
    """

    num_steps_per_env = 48
    max_iterations = 1200
    save_interval = 50
    experiment_name = "robotdog_teacher_velocity_tracking"
    run_name = "robotdog_teacher_v2_wide_scan_omnidirectional"

    clip_actions = 1.0

    # Teacher actor sees proprioception + privileged oracle + noisy local scan.
    # Critic additionally sees simulator-privileged contact/force signals.
    obs_groups = {
        "actor": ["policy", "teacher"],
        "critic": ["policy", "teacher", "privileged"],
    }
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            # Match the R16 PPO exploration initialization.
            init_std=0.40,
        ),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.016,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        # Match the adaptive schedule used to produce the R16 checkpoint.
        schedule="adaptive",
        # Standard discounted objective for per-step velocity tracking.
        gamma=0.99,
        lam=0.95,
        desired_kl=0.010,
        max_grad_norm=1.0,
    )
