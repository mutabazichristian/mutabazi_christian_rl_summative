import time

import matplotlib.pyplot as plt
import numpy as np
import pygame

from environment.custom_env import WorkplaceEnv
from environment.rendering import GameVisualization
from training.dqn_training import train_agents

print("Workplace Agent Simulation")
print("=" * 40)

print("1. Visual Demo")
print("2. Train and Compare Agents")
print("3. Quick Environment Test")
print("4. Play Trained Agents")

choice = input("Choose option (1-4): ")


def analyze_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Average rewards
    avg_rewards = {
        agent: np.mean(data["rewards"]) for agent, data in results.items()
    }
    axes[0, 0].bar(avg_rewards.keys(), avg_rewards.values())
    axes[0, 0].set_title("Average Reward per Episode")
    axes[0, 0].set_ylabel("Reward")

    # Survival rates
    survival_rates = {
        agent: np.mean([1 if t >= 480 else 0 for t in data["survival_times"]])
        for agent, data in results.items()
    }
    axes[0, 1].bar(survival_rates.keys(), survival_rates.values())
    axes[0, 1].set_title("Survival Rate (Complete 8 hours)")
    axes[0, 1].set_ylabel("Success Rate")

    # Average survival time
    avg_survival = {
        agent: np.mean(data["survival_times"])
        for agent, data in results.items()
    }
    axes[1, 0].bar(avg_survival.keys(), avg_survival.values())
    axes[1, 0].set_title("Average Survival Time")
    axes[1, 0].set_ylabel("Minutes")

    # Final trust points
    avg_trust = {
        agent: np.mean(data["trust_points"]) for agent, data in results.items()
    }
    axes[1, 1].bar(avg_trust.keys(), avg_trust.values())
    axes[1, 1].set_title("Average Final Trust Points")
    axes[1, 1].set_ylabel("Trust Points")

    plt.tight_layout()
    plt.show()

    # Print detailed analysis
    print("\nPERFORMANCE ANALYSIS")
    print("=" * 40)
    for agent_name, data in results.items():
        print(f"\n{agent_name.upper()} Agent:")
        print(
            f"  Average Reward: {np.mean(data['rewards']):.2f} ± {np.std(data['rewards']):.2f}"
        )
        print(f"  Survival Rate: {survival_rates[agent_name]:.2%}")
        print(
            f"  Average Survival Time: {avg_survival[agent_name]:.1f} minutes"
        )
        print(f"  Average Final Trust: {avg_trust[agent_name]:.1f} points")


def run_visual_demo():
    env = WorkplaceEnv()
    viz = GameVisualization(env)

    obs, _ = env.reset()
    clock = pygame.time.Clock()
    running = True

    print("Visual Demo")
    print("Controls:")
    print("  SPACE: Take random action")
    print("  ESC: Quit demo")
    print("  Close window: Exit")

    while (
        running
        and env.current_time < env.TOTAL_MINUTES
        and env.trust_points > 0
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Take random action
                    action = env.action_space.sample()
                    obs, reward, done, _, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.1f}, Trust: {info['trust_points']}")
                    if done:
                        print("Episode finished!")
                        break

        viz.render()
        clock.tick(30)

    viz.close()


def play_trained_agents():
    import os
    from stable_baselines3 import PPO, DQN
    import torch
    from training.pg_training import PolicyGradient
    from environment.rendering import GameVisualization

    print("Playing Trained Agents")
    print("=" * 40)

    env = WorkplaceEnv()
    viz = GameVisualization(env)


    models_available = []
    if os.path.exists("models/ppo_workplace_agent.zip"):
        models_available.append("PPO")
    if os.path.exists("models/dqn_workplace_agent.zip"):
        models_available.append("DQN")
    if os.path.exists("models/pg_workplace_agent.pth"):
        models_available.append("Policy Gradient")

    if not models_available:
        print("No trained models found. Please run training first (option 2).")
        return

    print(f"Available trained models: {', '.join(models_available)}")

    for agent_name in models_available:
        print(f"\nPlaying {agent_name} Agent...")
        print("Press Ctrl+C to skip to next agent or quit")

        try:
            if agent_name == "PPO":
                model = PPO.load("models/ppo_workplace_agent")
            elif agent_name == "DQN":
                model = DQN.load("models/dqn_workplace_agent")
            elif agent_name == "Policy Gradient":
                checkpoint = torch.load("models/pg_workplace_agent.pth", weights_only=False)
                model = PolicyGradient(checkpoint['state_dim'], checkpoint['action_dim'])
                model.network.load_state_dict(checkpoint['model_state_dict'])
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            obs, _ = env.reset()
            total_reward = 0
            clock = pygame.time.Clock()

            print(f"Starting episode with {agent_name}...")
            print("Press SPACE for faster playback, ESC to skip, or close window to stop")

            running = True
            speed_multiplier = 1

            while running and env.current_time < env.TOTAL_MINUTES and env.trust_points > 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        elif event.key == pygame.K_SPACE:
                            speed_multiplier = 5 if speed_multiplier == 1 else 1
                            print(f"Speed: {'5x' if speed_multiplier == 5 else '1x'}")

                if not running:
                    break

                if agent_name in ["PPO", "DQN"]:
                    action, _ = model.predict(obs, deterministic=True)
                else:  # Policy Gradient
                    action = model.select_action(obs)

                obs, reward, done, _, info = env.step(action)
                total_reward += reward

                viz.render()
                clock.tick(30 * speed_multiplier)

                if done:
                    break

            print(f"\n{agent_name} Episode Results:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Survival Time: {env.current_time} minutes")
            print(f"  Final Trust: {info['trust_points']} points")
            print(f"  Tasks Completed: {info['completed_tasks']}")
            print(f"  Tasks Failed: {info['failed_tasks']}")

        except KeyboardInterrupt:
            print(f"\nSkipping {agent_name}...")
            continue
        except Exception as e:
            print(f"Error playing {agent_name}: {e}")
            continue

    viz.close()


if choice == "1":
    run_visual_demo()
elif choice == "2":
    print("Training agents... This may take a few mintues.")
    results = train_agents()
    analyze_results(results)
elif choice == "4":
    play_trained_agents()
else:
    env = WorkplaceEnv(render_mode="human")
    obs, _ = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()

        if done:
            print(f"Episode ended. Final trust: {info['trust_points']}")
            break

    time.sleep(0.05)
