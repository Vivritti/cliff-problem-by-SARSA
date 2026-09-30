import numpy as np
import random
import gymnasium as gym

class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) Agent for discrete state-action environments.
    
    SARSA is an On-Policy Temporal Difference (TD) control algorithm that updates state-action 
    values using the action actually selected by the current policy.
    """
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.5, gamma: float = 0.99, epsilon: float = 0.1):
        """
        Initialize SARSA Agent.
        
        Parameters:
            n_states (int): Total number of states in the environment.
            n_actions (int): Total number of possible actions per state.
            alpha (float): Learning rate (default: 0.5).
            gamma (float): Discount factor for future rewards (default: 0.99).
            epsilon (float): Exploration rate for epsilon-greedy policy (default: 0.1).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.Q = np.zeros((n_states, n_actions))

    def get_action(self, state: int) -> int:
        """
        Select an action using the Epsilon-Greedy policy strategy.
        
        Parameters:
            state (int): Current state index.
            
        Returns:
            int: Selected action index.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # Greedy action choice (break ties randomly if needed)
            max_val = np.max(self.Q[state])
            best_actions = np.where(self.Q[state] == max_val)[0]
            return random.choice(best_actions)

    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool) -> None:
        """
        Update the Q-value using the SARSA update rule:
        Q(S, A) <- Q(S, A) + alpha * [Reward + gamma * Q(S', A') - Q(S, A)]
        
        Parameters:
            state (int): Current state.
            action (int): Action taken in current state.
            reward (float): Reward received.
            next_state (int): Next state reached.
            next_action (int): Action selected in next state.
            done (bool): Whether the episode terminated/truncated at next_state.
        """
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state, next_action]
            
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(self, env: gym.Env, episodes: int = 500) -> tuple[list[float], list[int]]:
        """
        Train the agent over a specified number of episodes.
        
        Parameters:
            env (gym.Env): Gymnasium environment (e.g. CliffWalking-v1).
            episodes (int): Total training episodes.
            
        Returns:
            tuple: (total_rewards_per_episode, episode_lengths)
        """
        rewards_history = []
        lengths_history = []

        for episode in range(episodes):
            state, _ = env.reset()
            action = self.get_action(state)

            total_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_action = self.get_action(next_state)

                # Update Q table
                self.update(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action

                total_reward += reward
                episode_length += 1

            rewards_history.append(total_reward)
            lengths_history.append(episode_length)

        return rewards_history, lengths_history

    def evaluate(self, env: gym.Env, max_steps: int = 200) -> tuple[float, int]:
        """
        Evaluate the learned policy deterministically (greedy action selection).
        
        Parameters:
            env (gym.Env): Gymnasium environment.
            max_steps (int): Maximum step limit.
            
        Returns:
            tuple: (total_reward, episode_length)
        """
        state, _ = env.reset()
        total_reward = 0.0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            # Deterministic greedy action
            action = int(np.argmax(self.Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_length += 1

        return total_reward, episode_length
