import numpy as np
import matplotlib.pyplot as plt
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent

def run_repetition(agent_type, n_rep, n_episodes, n=0, epsilon=0.1, alpha=0.1, gamma=1.0):
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
    mean_returns = run_repetition("qlearning", 1, 10000)
    plt.plot(mean_returns)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Q-Learning with 10000 episodes")
    plt.savefig("Q-learning_10000.png")
    plt.close()

    mean_returns = run_repetition("qlearning", 100, 1000)
    plt.plot(mean_returns)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Q-Learning with 100 repetitions and 1000 episodes")
    plt.savefig("Q-learning_100x1000.png")
    plt.close()

    alpha_values = [0.01, 0.1, 0.5, 0.9]
    for alpha in alpha_values:
        mean_returns = run_repetition('qlearning', 100, 1000, alpha=alpha)
        plt.plot(mean_returns, label=f"alpha={alpha}")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Q-learning with different alpha values")
    plt.legend()
    plt.savefig("Q-learning_alpha_values.png")
    plt.close()

    # Q-learning greedy paths
    env = ShortcutEnvironment()
    agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1, alpha=0.5, gamma=1.0)
    agent.train(env, 10000)
    env.render_greedy(agent.Q)

    ''' SARSA '''
    mean_returns = run_repetition('sarsa', 1, 10000)
    plt.plot(mean_returns)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("SARSA with 10000 episodes")
    plt.savefig("SARSA_10000.png")
    plt.close()

    mean_returns = run_repetition('sarsa', 100, 1000)
    plt.plot(mean_returns)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("SARSA with 100 repetitions and 1000 episodes")
    plt.savefig("SARSA_100x1000.png")
    plt.close()

    alpha_values = [0.01, 0.1, 0.5, 0.9]
    for alpha in alpha_values:
        mean_returns = run_repetition('sarsa', 100, 1000, alpha=alpha)
        plt.plot(mean_returns, label=f"alpha={alpha}")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("SARSA with different alpha values")
    plt.legend()
    plt.savefig("SARSA_alpha_values.png")
    plt.close()

    # SARSA greedy paths
    env = ShortcutEnvironment()
    agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1, alpha=0.5, gamma=1.0)
    agent.train(env, 10000)
    env.render_greedy(agent.Q)

    ''' Stormy Weather '''
    env = WindyShortcutEnvironment()
    agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1, alpha=0.1, gamma=1.0)
    agent.train(env, 10000)
    env.render_greedy(agent.Q)

    env = WindyShortcutEnvironment()
    agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1, alpha=0.1, gamma=1.0)
    agent.train(env, 10000)
    env.render_greedy(agent.Q)

    ''' Expected SARSA '''
    mean_returns = run_repetition("expectedsarsa", 1, 10000)
    plt.plot(mean_returns)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Expected SARSA with 10000 episodes")
    plt.savefig("Expected_SARSA_10000.png")
    plt.close()

    mean_returns = run_repetition("expectedsarsa", 100, 1000)
    plt.plot(mean_returns)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Expected SARSA with 100 repetitions and 1000 episodes")
    plt.savefig("Expected_SARSA_100x1000.png")
    plt.close()

    alpha_values = [0.01, 0.1, 0.5, 0.9]
    for alpha in alpha_values:
        mean_returns = run_repetition('expectedsarsa', 100, 1000, alpha=alpha)
        plt.plot(mean_returns, label=f"alpha={alpha}")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Expected SARSA with different alpha values")
    plt.legend()
    plt.savefig("Expected_SARSA_alpha_values.png")
    plt.close()

    # Expected SARSA greedy paths
    env = ShortcutEnvironment()
    agent = ExpectedSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1, alpha=0.5, gamma=1.0) # maybe change alpha after analyzing the results
    agent.train(env, 10000)
    env.render_greedy(agent.Q)

    ''' n-step SARSA '''
    mean_returns = run_repetition('nstepsarsa', 1, 10000, n=2)
    plt.plot(mean_returns)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("n-step SARSA with 10000 episodes and n=2")
    plt.savefig("n-step_SARSA_10000.png")
    plt.close()

    n_values = [1, 2, 5, 10, 25]
    for n in n_values:
        mean_returns = run_repetition('sarsa', 100, 1000, n=n)
        plt.plot(mean_returns, label=f"n={n}")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("n-step SARSA with different n values")
    plt.legend()
    plt.savefig("n-step_SARSA_n_values.png")
    plt.close()

    ''' Comparison '''
    qlearning_returns = run_repetition("qlearning", 100, 1000, alpha=0.5)
    sarsa_returns = run_repetition("sarsa", 100, 1000, alpha=0.5)
    expectedsarsa_returns = run_repetition("expectedsarsa", 100, 1000, alpha=0.5) # maybe change alpha after analyzing the results
    nstepsarsa_returns = run_repetition("nstepsarsa", 100, 1000, n=2) # maybe change n and alpha after analyzing the results

    plt.plot(qlearning_returns, label="Q-learning")
    plt.plot(sarsa_returns, label="SARSA")
    plt.plot(expectedsarsa_returns, label="Expected SARSA")
    plt.plot(nstepsarsa_returns, label="n-step SARSA")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Comparison of Q-learning, SARSA, Expected SARSA, and n-step SARSA")
    plt.legend()
    plt.savefig("Comparison.png")
    plt.close()