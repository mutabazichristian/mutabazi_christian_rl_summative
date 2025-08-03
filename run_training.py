#!/usr/bin/env python3

import time
import json
import matplotlib.pyplot as plt
import numpy as np

from environment.custom_env import WorkplaceEnv
from training.dqn_training import train_agents


def analyze_results(results):
    """Analyze and visualize results"""
    print("  Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))


    avg_rewards = {
        agent: np.mean(data["rewards"]) for agent, data in results.items()
    }
    axes[0, 0].bar(avg_rewards.keys(), avg_rewards.values())
    axes[0, 0].set_title("Average Reward per Episode")
    axes[0, 0].set_ylabel("Reward")


    survival_rates = {
        agent: np.mean([1 if t >= 480 else 0 for t in data["survival_times"]])
        for agent, data in results.items()
    }
    axes[0, 1].bar(survival_rates.keys(), survival_rates.values())
    axes[0, 1].set_title("Survival Rate (Complete 8 hours)")
    axes[0, 1].set_ylabel("Success Rate")


    avg_survival = {
        agent: np.mean(data["survival_times"])
        for agent, data in results.items()
    }
    axes[1, 0].bar(avg_survival.keys(), avg_survival.values())
    axes[1, 0].set_title("Average Survival Time")
    axes[1, 0].set_ylabel("Minutes")


    avg_trust = {
        agent: np.mean(data["trust_points"]) for agent, data in results.items()
    }
    axes[1, 1].bar(avg_trust.keys(), avg_trust.values())
    axes[1, 1].set_title("Average Final Trust Points")
    axes[1, 1].set_ylabel("Trust Points")

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("  ✓ Results plot saved as 'training_results.png'")

    # Print detailed analysis
    print("  ✓ Generating performance analysis...")
    print("\n=== PERFORMANCE ANALYSIS ===")
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


def main():
    print("Workplace Agent Training")
    print("=" * 40)
    print("Training agents... This may take a few minutes.")

    start_time = time.time()

    try:
        results = train_agents()

        end_time = time.time()
        training_time = end_time - start_time

        print(f"\nTraining completed in {training_time:.1f} seconds")

        results_with_metadata = {
            'training_time': training_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }

        with open('training_results.json', 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        print("  results saved to training_results.json")

        print("\nAnalyzing results...")
        analyze_results(results)

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed!")
        exit(1)
