"""Phase 5: PPO learns to balance the cartpole.

Run:  uv run python scripts/08_ppo_cartpole.py
"""
import robolab
from robolab.envs.cartpole import CartpoleEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


XML = str(robolab.ASSETS_DIR / "cartpole.xml")


def make_env():
    # A fresh CartpoleEnv per parallel worker. init_angle=0.2 makes the task
    # non-trivial (the pole starts meaningfully off-upright).
    return CartpoleEnv(XML, init_angle=0.2)


def main():
    # 8 environments stepped in parallel -> 8x the experience per wall-clock second.
    vec_env = make_vec_env(make_env, n_envs=8)

    model = PPO(
        policy="MlpPolicy",     # small feedforward net (default: 2 hidden layers x 64, tanh)
        env=vec_env,
        n_steps=1024,           # steps collected PER env before each update (batch = 8*1024)
        batch_size=256,         # minibatch size for the SGD passes
        n_epochs=10,            # how many passes over each collected batch
        gamma=0.99,             # discount: how much future reward counts now
        gae_lambda=0.95,        # GAE: bias/variance knob for advantage estimation
        clip_range=0.2,         # the "proximal" clip — caps how far each update moves the policy
        ent_coef=0.0,           # entropy bonus (exploration); 0 is fine for cartpole
        learning_rate=3e-4,     # Adam step size
        verbose=1,              # print the training table
        device="cpu",           # small MLP -> CPU is faster than GPU here (see chat notes)
        tensorboard_log="outputs/tb_cartpole",
    )

    model.learn(total_timesteps=200_000)
    model.save("outputs/ppo_cartpole")

    # Evaluate the trained policy on a fresh env (no parallelism).
    eval_env = make_env()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Eval over 10 episodes: mean_reward = {mean_reward:.1f} +/- {std_reward:.1f}  (max 1000)")


if __name__ == "__main__":
    main()
