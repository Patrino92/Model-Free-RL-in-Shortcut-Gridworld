# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.

import numpy as np
import matplotlib.pyplot as plt
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent

def run_repetition(agent_type, n_rep, n_episodes, epsilon=0.1, alpha=0.1, gamma=1.0):
    total_returns = []

    for rep in range(n_rep):
        env = ShortcutEnvironment()
        if agent_type == 'qlearning':
            agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma)
        elif agent_type == 'sarsa':
            agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma)
        
        episode_returns = agent.train(env, n_episodes)
        total_returns.append(episode_returns)

    return np.mean(total_returns, axis=0)

if __name__ == "__main__":
    # Q-learning
    
    #SARSA
    #mean_returns = run_repetition('sarsa', 1, 10000)
    # mean_returns = run_repetition('sarsa', 100, 1000)
    # plt.plot(mean_returns)
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.title("SARSA")
    # plt.show()
    
    alpha = [0.01, 0.1, 0.5, 0.9]
    for a in alpha:
        mean_returns = run_repetition('sarsa', 100, 1000, alpha=a)
        plt.plot(mean_returns, label=f"alpha={a}")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("SARSA with different alpha")
    plt.legend()
    plt.show()