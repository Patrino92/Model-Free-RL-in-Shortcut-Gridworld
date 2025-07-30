# Model-Free RL in Shortcut Gridworld
**Assignment Year: 2024-2025**

This project explores four model-free reinforcement learning algorithms: **Q-Learning**, **SARSA**, **Expected SARSA**, and **n-step SARSA**. Each method is implemented and evaluated in a custom 12×12 grid-world called the **Shortcut Environment**, which introduces challenges such as cliffs and stochastic wind. We analyze learning curves, parameter sensitivity, and policy behaviors across methods.

## Overview
The **ShortcutEnvironment** simulates a grid-based navigation task:
- The agent starts from one of two positions (randomly selected).
- The goal is to reach a designated target state.
- Certain grid cells act as cliffs: entering them resets the agent and incurs a large penalty.
- All actions incur a small negative reward unless the agent reaches the goal.
- A **WindyShortcutEnvironment** variant introduces a 50% chance of being pushed downward after any action.

We implement and test the following RL agents:

- **Q-Learning**  
- **SARSA**  
- **Expected SARSA**  
- **n-step SARSA**

## Features
- Custom Shortcut and Windy environments.
- ε-greedy exploration and Q-table updates.
- Batch experiment runner with plotting.
- Visualizations of learned greedy policies.
- Comparisons across hyperparameters.

## Contributors
- **Kacper Nizielski**
- **Emmanouil Zagoriti**

## References
- Thomas Moerland, part of Introduction to Reinforcement Learning course, Leiden University
  
