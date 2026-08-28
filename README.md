# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

### NAME : Sana Fathima H
### REG NO.: 212223240145

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm using the FrozenLake environment in Gymnasium. The agent learns an optimal policy by interacting with the environment, selecting actions using an epsilon-greedy strategy, and updating the Q-values based on the received rewards and the next state-action pair.


## Software Requirements

Operating System: Windows / Linux / macOS

Programming Language: Python 3.x

IDE/Platform: Jupyter Notebook / Google Colab / VS Code

Libraries:

Gymnasium – for the FrozenLake environment

NumPy – for Q-table and numerical calculations

Matplotlib – for plotting the SARSA learning curve

Environment: FrozenLake-v1

Algorithm: SARSA (On-Policy Temporal Difference Learning)

## Environment Description
```
env = gym.make(
    "FrozenLake-v1",
    desc=[
        "FFSF",
        "FHFH",
        "FFFH",
        "HGFF"
    ],
    is_slippery=True
)
```
## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm

1.Initialize the Q-table with zeros.

2.Set epsilon = 1.0 for initial exploration.

3.Reset the FrozenLake environment and obtain the initial state S.

4.Select action A using the epsilon-greedy policy.

5.Take action A and observe reward R and next state S'.

6.Select the next action A' using the epsilon-greedy policy.

7.Update the Q-value using:
```
Q(S,A)←Q(S,A)+α[R+(not terminated)γQ(S′,A′)−Q(S,A)]

```

8.Set S = S' and A = A'.

9.Repeat until the episode terminates.

10.Apply epsilon decay:

```
epsilon = max(epsilon_min, epsilon * epsilon_decay)

```
11.Repeat the process for the specified number of episodes.

12.Extract the state-value function and learned policy from the final Q-table.

## Python Program

```python

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------
# Write your code here

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0
    terminated = False
    truncated = False

    for step in range(max_steps_per_episode):

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # SARSA update
        if not terminated and not truncated:

            next_action = epsilon_greedy_action(next_state, epsilon)

            target = reward + gamma * Q[next_state, next_action]

            Q[state, action] = Q[state, action] + alpha * (
                target - Q[state, action]
            )

            state = next_state
            action = next_action

        else:
            # Terminal state
            target = reward

            Q[state, action] = Q[state, action] + alpha * (
                target - Q[state, action]
            )

            break

    episode_rewards.append(total_reward)

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

```

## Output

Final Q-table:

<img width="367" height="398" alt="image" src="https://github.com/user-attachments/assets/502916f3-dd6b-485f-9a88-e1a2047a2f6a" />


Estimated State-Value Function:


<img width="393" height="133" alt="image" src="https://github.com/user-attachments/assets/f1ac9176-0000-4e4f-bb4f-974a6d803b26" />


Learned Policy:

<img width="280" height="132" alt="image" src="https://github.com/user-attachments/assets/0d74a14a-4364-44a6-b66b-83a6183bd367" />


Average reward over last 1000 episodes: 

<img width="522" height="38" alt="image" src="https://github.com/user-attachments/assets/6b1656d0-2c0e-4448-998f-f08c737708c0" />


Plot Learning Curve:

<img width="942" height="610" alt="image" src="https://github.com/user-attachments/assets/ae5d74fd-bb15-4da2-814a-d5859e237b8a" />



## Result

SARSA was successfully implemented in the FrozenLake environment.
The agent learned optimal Q-values and a policy to reach the goal while avoiding holes.
Epsilon decay improved learning by gradually changing from exploration to exploitation.

## Inference
With fixed epsilon, the exploration rate remains constant throughout training, so the agent continues to explore and exploit at the same ratio in every episode. With epsilon decay, the agent starts with high exploration and gradually reduces epsilon, allowing it to explore more in the beginning and exploit the learned policy more in later episodes. Therefore, epsilon decay generally helps the agent learn a better policy over time.


