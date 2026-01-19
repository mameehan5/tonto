import argparse
import os
from stable_baselines3.common.env_util import make_vec_env
# Use MaskablePPO to support action masks
from sb3_contrib.ppo_mask import MaskablePPO
# Maskable callback and evaluation
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.callbacks import CheckpointCallback



# Import the Tonto environment and config
from env.environment import TontoEnv

def make_env_fn(num_players, num_cards, num_decks, num_rounds, seed=None):
    """Return a function that creates a new CardGameEnv with given config."""
    def _init():
        env = TontoEnv(
            num_players=num_players, 
            num_decks=num_decks, 
            num_rounds=num_rounds, 
            num_cards=num_cards, 
            seed=seed
        )
        return ActionMasker(env, lambda e: e.action_masks())
    return _init

def main(args):
    # Create a vectorized environment (parallel instances for faster data collection)
    env_fn = make_env_fn(
        num_players=args.num_players,
        num_cards=args.num_cards,
        num_decks=args.num_decks,
        num_rounds=args.num_rounds,
        seed=args.seed,
    )
    vec_env = make_vec_env(
        env_id=env_fn,
        n_envs=args.num_envs, 
        # vec_env_cls=SubprocVecEnv # parallelize over envs
    )

    masks = get_action_masks(vec_env)
    assert masks is not None

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=50_000,
    #     save_path=args.save_path,
    #     name_prefix="ppo_tonto_checkpoint"
    # )

    # Configure PPO with MlpPolicy and optional GPU usage
    if args.load_file is None:
        model = MaskablePPO(
            "MultiInputPolicy", 
            vec_env,
            verbose=1,
            # tensorboard_log=args.log_path, 
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate, 
            gamma=0.995, # 0.99, higher for longer episodes
            gae_lambda=0.97, # increased from default since sparse rewards
            ent_coef=0.03, # increased from 0.01 to promote variability, less deterministic
            vf_coef=0.5, # default
            clip_range=0.1, # 0.2 default, lower to slow policy update 
            n_epochs=10, # 5-10, num epochs over rollout data update
            max_grad_norm=0.5, # default
            device="auto",  # use GPU if available
        )
    else:
        model = MaskablePPO.load(
            args.load_file,
            env=vec_env,
            device="auto"
        )

    # Optional: evaluation setup using a separate env
    if args.eval_freq > 0:
        eval_env = TontoEnv(
            num_players=args.num_players, 
            num_decks=args.num_decks, 
            num_rounds=args.num_rounds, 
            num_cards=args.num_cards, 
        )
        # Wrap eval_env for masking if needed (similar to training env)
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=args.save_path,
            log_path=args.log_path,
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False
        )
    else:
        eval_callback = None

    # Train the agent
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    # Save the final model
    model.save(os.path.join(args.save_path, "ppo_tonto_model"))

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(
        model, 
        vec_env, 
        n_eval_episodes=100, 
        deterministic=True
    )
    print(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_players", type=int, default=1,
                        help="Number of players in the game")
    parser.add_argument("--num_cards", type=int, default=4,
                        help="Cards per player (must be even)")
    parser.add_argument("--num_decks", type=int, default=1,
                        help="Number of card decks")
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="Rounds per episode")
    parser.add_argument("--num_envs", type=int, default=2,
                        help="Number of parallel environments for training")
    parser.add_argument("--total_timesteps", type=int, default=100000,
                        help="Total timesteps for training")
    parser.add_argument("--n_steps", type=int, default=8192,
                        help="Rollout steps per update (PPO hyperparameter)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Minibatch size (PPO hyperparameter)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--eval_freq", type=int, default=20000,
                        help="Evaluate every X timesteps (0 to disable)")
    parser.add_argument("--save_path", type=str, default="models/",
                        help="Directory to save models and logs")
    parser.add_argument("--load_file", type=str, default=None,
                        help="File to load model from checkpoint")
    parser.add_argument("--log_path", type=str, default="logs/",
                        help="Directory for evaluation logs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    main(args)
