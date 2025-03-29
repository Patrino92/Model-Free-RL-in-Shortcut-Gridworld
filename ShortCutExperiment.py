# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import matplotlib.pyplot as plt
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent
import sys
sys.stdout.reconfigure(encoding='utf-8')


def run_repetition(agent_type, n_rep, n_episodes, n=0, epsilon=0.1, alpha=0.5, gamma=1.0):
    total_returns = []

    for _ in range(n_rep):
        env = ShortcutEnvironment()
        
        if agent_type == 'qlearning':
            agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma)
        elif agent_type == 'sarsa':
            agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma)
        elif agent_type == 'expectedsarsa':
            agent = ExpectedSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma)
        elif agent_type == 'nstepsarsa':
            agent = nStepSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), n=n, epsilon=epsilon, alpha=alpha, gamma=gamma)
       
        episode_returns = agent.train(env, n_episodes)
        total_returns.append(episode_returns)

    return np.mean(total_returns, axis=0)

if __name__ == "__main__":
    ''' Q-learning '''
    mean_returns = run_repetition("qlearning",1,10000)
    # mean_returns = run_repetition("qlearning", 100, 1000)
    plt.plot(mean_returns)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Q-Learning")
    plt.show()

    # alpha = [0.01, 0.1, 0.5, 0.9]
    # for a in alpha:
    #     mean_returns = run_repetition('qlearning', 100, 1000, alpha=a)
    #     plt.plot(mean_returns, label=f"alpha={a}")
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.title("Q-learning with different alpha values")
    # plt.legend()
    # plt.show()
    ''' SARSA '''
    # mean_returns = run_repetition('sarsa', 1, 10000)
    # mean_returns = run_repetition('sarsa', 100, 1000)
    # plt.plot(mean_returns)
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.title("SARSA")
    # plt.show()
    
    # alpha = [0.01, 0.1, 0.5, 0.9]
    # for a in alpha:
    #     mean_returns = run_repetition('sarsa', 100, 1000, alpha=a)
    #     plt.plot(mean_returns, label=f"alpha={a}")
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.title("SARSA with different alpha values")
    # plt.legend()
    # plt.show()

    ''' Stormy Weather '''
    # env = WindyShortcutEnvironment()
    # # agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1, alpha=0.1)
    # agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1, alpha=0.1)
    # agent.train(env, 10000)
    # env.render_greedy(agent.Q) # Does not show the path nicely

    ''' Expected SARSA '''
    # mean_returns = run_repetition("expectedsarsa",1,10000)
    # mean_returns = run_repetition("expectedsarsa", 100, 1000)
    # plt.plot(mean_returns)
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.title("Expected SARSA")
    # plt.show()

    # alpha = [0.01, 0.1, 0.5, 0.9]
    # for a in alpha:
    #     mean_returns = run_repetition('expectedsarsa', 100, 1000, alpha=a)
    #     plt.plot(mean_returns, label=f"alpha={a}")
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.title("Expected SARSA with different alpha values")
    # plt.legend()
    # plt.show()

    ''' n-step SARSA '''
    # mean_returns = run_repetition('nstepsarsa', 1, 10000, n=2)
    # plt.plot(mean_returns)
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.title("SARSA")
    # plt.show()

    # n_values = [1, 2, 5, 10, 25]
    # for n in n_values:
    #     mean_returns = run_repetition('sarsa', 100, 1000, n=n)
    #     plt.plot(mean_returns, label=f"n={n}")
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.title("n-step SARSA with different n values")
    # plt.legend()
    # plt.show()