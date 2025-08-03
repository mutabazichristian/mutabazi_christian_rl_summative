import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from environment.custom_env import WorkplaceEnv
from training.pg_training import PolicyGradient


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, agent_name, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.agent_name = agent_name
        self.last_print_time = time.time()
        self.print_interval = 30

    def _on_step(self) -> bool:
        current_time = time.time()
        if current_time - self.last_print_time > self.print_interval:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            print(f"  {self.agent_name} training: {progress:.1f}% done ({self.num_timesteps}/{self.total_timesteps})")
            self.last_print_time = current_time
        return True


def train_agents():
    env = WorkplaceEnv()

    EPISODES = 2000
    TOTAL_TIMESTEPS = EPISODES * 480

    print(f"Training config: {EPISODES} episodes, {TOTAL_TIMESTEPS:,} timesteps")

    results = {
        "ppo": {"rewards": [], "survival_times": [], "trust_points": []},
        "dqn": {"rewards": [], "survival_times": [], "trust_points": []},
        "pg": {"rewards": [], "survival_times": [], "trust_points": []},
    }

    print("\nTraining PPO agent...")
    print("  setting up model...")
    start_time = time.time()

    vec_env = DummyVecEnv([lambda: env])
    ppo_model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=0.0001,
        n_steps=4096,
        batch_size=256,
        device="cpu",
    )

    ppo_callback = ProgressCallback(TOTAL_TIMESTEPS, "PPO")
    print(f"  starting PPO training ({TOTAL_TIMESTEPS:,} timesteps)...")
    ppo_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ppo_callback)

    ppo_model.save("models/ppo_workplace_agent")
    print(f"  PPO model saved")

    ppo_train_time = time.time() - start_time
    print(f"  PPO training done in {ppo_train_time:.1f} seconds")

    print("  testing PPO agent (100 episodes)...")
    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        results["ppo"]["rewards"].append(total_reward)
        results["ppo"]["survival_times"].append(env.current_time)
        results["ppo"]["trust_points"].append(info["trust_points"])

        if (episode + 1) % 25 == 0:
            print(f"    PPO test episode {episode + 1}/100 done")

    print("\nTraining DQN agent...")
    print("  setting up DQN model...")
    start_time = time.time()
    dqn_model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=0.0003,
        buffer_size=100000,
        exploration_fraction=0.4,
        device="cpu",
    )

    dqn_callback = ProgressCallback(TOTAL_TIMESTEPS, "DQN")
    print(f"  starting DQN training ({TOTAL_TIMESTEPS:,} timesteps)...")
    dqn_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=dqn_callback)

    dqn_model.save("models/dqn_workplace_agent")
    print(f"  DQN model saved")

    dqn_train_time = time.time() - start_time
    print(f"  DQN training done in {dqn_train_time:.1f} seconds")

    print("  testing DQN agent (100 episodes)...")
    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = dqn_model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        results["dqn"]["rewards"].append(total_reward)
        results["dqn"]["survival_times"].append(env.current_time)
        results["dqn"]["trust_points"].append(info["trust_points"])

        if (episode + 1) % 25 == 0:
            print(f"    DQN test episode {episode + 1}/100 done")

    print("\nTraining Policy Gradient agent...")
    start_time = time.time()
    pg_agent = PolicyGradient(
        env.observation_space.shape[0], env.action_space.n, lr=0.002
    )

    print(f"  starting Policy Gradient training ({EPISODES} episodes)...")
    for episode in range(EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = pg_agent.select_action(obs)
            obs, reward, done, _, info = env.step(action)
            pg_agent.rewards.append(reward)
            total_reward += reward

        pg_agent.update()

        if (episode + 1) % 100 == 0:
            progress = ((episode + 1) / EPISODES) * 100
            print(f"  Policy Gradient: {progress:.1f}% done ({episode + 1}/{EPISODES} episodes)")

        if episode >= EPISODES - 100:  # Last 100 episodes for testing
            results["pg"]["rewards"].append(total_reward)
            results["pg"]["survival_times"].append(env.current_time)
            results["pg"]["trust_points"].append(info["trust_points"])

    import torch
    torch.save({
        'model_state_dict': pg_agent.network.state_dict(),
        'optimizer_state_dict': pg_agent.optimizer.state_dict(),
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.n
    }, "models/pg_workplace_agent.pth")
    print(f"  Policy Gradient model saved")

    pg_train_time = time.time() - start_time
    print(f"  Policy Gradient training done in {pg_train_time:.1f} seconds")

    print(f"\nAll training completed!")
    print(f"  Total time: {(ppo_train_time + dqn_train_time + pg_train_time):.1f} seconds")
    print(f"  Models saved in models/ directory")

    return results
