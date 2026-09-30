import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from sarsa import SARSAAgent

def plot_training_results(rewards: list[float], lengths: list[int], save_path: str = "assets/training_rewards.png"):
    """
    Plot reward progress and episode lengths with moving average smoothing.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate moving averages (window = 20)
    window = 20
    rewards_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    lengths_ma = np.convolve(lengths, np.ones(window)/window, mode='valid')
    
    # Plot Total Rewards
    ax1.plot(rewards, alpha=0.3, color='steelblue', label='Raw Episode Reward')
    ax1.plot(range(window - 1, len(rewards)), rewards_ma, color='darkblue', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax1.set_title("Training Reward per Episode (SARSA)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot Episode Lengths
    ax2.plot(lengths, alpha=0.3, color='coral', label='Raw Episode Length')
    ax2.plot(range(window - 1, len(lengths)), lengths_ma, color='firebrick', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax2.set_title("Episode Length per Episode (SARSA)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps to Reach Goal")
    ax2.set_grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training performance plot to '{save_path}'")

def plot_policy_grid(Q: np.ndarray, shape: tuple[int, int] = (4, 12), save_path: str = "assets/policy_grid.png"):
    """
    Visualize the learned policy on the Cliff Walking grid map.
    Grid shape: 4 rows x 12 columns
    Actions: 0: Up (^), 1: Right (>), 2: Down (v), 3: Left (<)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    grid = np.zeros(shape)
    
    # Mark cliff tiles and start/goal
    # Start: (3, 0) -> state 36
    # Cliff: (3, 1..10) -> states 37..46
    # Goal: (3, 11) -> state 47
    
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(-0.5, shape[0] - 0.5)
    ax.invert_yaxis()  # top row is 0
    
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))
    ax.set_grid(True, color='black', linestyle='-', linewidth=1)
    
    for r in range(shape[0]):
        for c in range(shape[1]):
            state = r * shape[1] + c
            
            if r == 3 and 1 <= c <= 10:
                # Cliff
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='red', alpha=0.6)
                ax.add_patch(rect)
                ax.text(c, r, "CLIFF", ha='center', va='center', fontweight='bold', color='white', fontsize=9)
            elif r == 3 and c == 0:
                # Start
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='lightgreen', alpha=0.6)
                ax.add_patch(rect)
                best_action = np.argmax(Q[state])
                ax.text(c, r, f"S\n{action_symbols[best_action]}", ha='center', va='center', fontweight='bold', fontsize=12)
            elif r == 3 and c == 11:
                # Goal
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='gold', alpha=0.7)
                ax.add_patch(rect)
                ax.text(c, r, "GOAL\n🚩", ha='center', va='center', fontweight='bold', fontsize=11)
            else:
                # Normal State
                best_action = np.argmax(Q[state])
                ax.text(c, r, action_symbols[best_action], ha='center', va='center', fontsize=16, fontweight='bold', color='darkblue')

    ax.set_title("Learned SARSA Policy Grid (Cliff Walking)", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved policy grid visualization to '{save_path}'")

def main():
    parser = argparse.ArgumentParser(description="SARSA Algorithm for Gymnasium CliffWalking-v1")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate (epsilon)")
    parser.add_argument("--render", action="store_true", help="Render final evaluation step in human mode")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("      SARSA Reinforcement Learning: Cliff Walking")
    print("=" * 60)
    print(f"Hyperparameters: episodes={args.episodes}, alpha={args.alpha}, gamma={args.gamma}, epsilon={args.epsilon}")
    
    # Create Gym Environment
    env = gym.make("CliffWalking-v1")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize Agent
    agent = SARSAAgent(n_states, n_actions, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    
    print("\nTraining SARSA Agent...")
    rewards, lengths = agent.train(env, episodes=args.episodes)
    print("Training Complete!")
    
    # Summary Metrics
    print(f"\nLast 50 Episodes Average Reward: {np.mean(rewards[-50:]):.2f}")
    print(f"Last 50 Episodes Average Length: {np.mean(lengths[-50:]):.2f}")
    
    # Save Plots
    plot_training_results(rewards, lengths, "assets/training_rewards.png")
    plot_policy_grid(agent.Q, (4, 12), "assets/policy_grid.png")
    
    # Evaluation
    eval_env = gym.make("CliffWalking-v1", render_mode="human" if args.render else None)
    eval_reward, eval_steps = agent.evaluate(eval_env)
    print(f"\nEvaluation Results (Greedy Policy):")
    print(f"  Total Reward: {eval_reward}")
    print(f"  Total Steps : {eval_steps}")
    eval_env.close()

if __name__ == "__main__":
    main()
