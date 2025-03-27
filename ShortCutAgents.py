import numpy as np #IT WAS NOT INCLUDED

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        p = np.random.rand()
        if p < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state, :])
        return action
        
    def update(self, state, action, reward, done, next_state): # Augment arguments if necessary
        # TO DO: Implement Q-learning update
        if done:
            self.Q[state, action] = self.Q[state, action] + self.alpha * (reward - self.Q[state, action])
        else:
            self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

    
    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the cumulative reward (=return) per episode
        episode_returns = []
        for _ in range(n_episodes):
            env.reset()
            state = env.state()
            episode_reward = 0
            while not env.done():
                action = self.select_action(state)
                reward = env.step(action)
                episode_reward += reward
                self.update(state, action, reward, env.done(), env.state())
                state = env.state()
            episode_returns.append(episode_reward)
        return episode_returns

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        p = np.random.rand()
        if p < self.epsilon:
          action = np.random.randint(self.n_actions)
        else:
          action = np.argmax(self.Q[state, :])
        return action
        
    def update(self, state, action, next_state, next_action, reward, done): # Augment arguments if necessary
        # TO DO: Implement SARSA update
        if done:
            self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
        else:
            self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])

    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode\
        episode_returns = []

        for _ in range(n_episodes):
            env.reset()
            state = env.state()
            action = self.select_action(state)
            episode_return = 0

            while not env.done():
                reward = env.step(action)
                next_state = env.state()
                next_action = self.select_action(next_state)
                self.update(state, action, next_state, next_action, reward, env.done())

                state = next_state
                action = next_action
                episode_return += reward

            episode_returns.append(episode_return)
        return episode_returns


class ExpectedSARSAAgent(object):


    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, state, action, reward, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        pass

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns    


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        p = np.random.rand()
        if p < self.epsilon:
          action = np.random.randint(self.n_actions)
        else:
          action = np.argmax(self.Q[state, :])
        return action
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        G = 0
        for i in range(len(rewards)):
            G += self.gamma ** i * rewards[i]
        if not done:
            G += self.gamma ** len(rewards) * self.Q[states[-1], actions[-1]]
        self.Q[states[0], actions[0]] += self.alpha * (G - self.Q[states[0], actions[0]])
    
    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for _ in range(n_episodes):
            env.reset()
            state = env.state()
            action = self.select_action(state)
            episode_return = 0

            states, actions, rewards = [], [], []
            while not env.done(): 
                reward = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = env.state()
                action = self.select_action(state)
                episode_return += reward 

                # Update state-action pairs that have n steps before episode ends
                if len(actions) > self.n:
                    self.update(states, actions, rewards, env.done())
                    states.pop(0)
                    actions.pop(0)
                    rewards.pop(0)
            
            # Update state-action pairs that do not have n steps before episode ends
            while len(actions) > 0:
                self.update(states, actions, rewards, env.done())
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)
            
            episode_returns.append(episode_return)
        return episode_returns