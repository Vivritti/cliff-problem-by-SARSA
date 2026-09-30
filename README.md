# 🧗 Cliff Walking using SARSA Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end Python & Jupyter implementation of the **SARSA (State-Action-Reward-State-Action)** algorithm applied to the classic **Cliff Walking** environment (`CliffWalking-v1`) from Farama Gymnasium.

---

## 📌 Problem Overview

**Cliff Walking** is a standard gridworld environment introduced in Sutton & Barto's *Reinforcement Learning: An Introduction*.

- **Grid Dimensions**: $4 \times 12$ (48 discrete states)
- **Start State**: Bottom-left corner (State 36 / `[3, 0]`)
- **Goal State**: Bottom-right corner (State 47 / `[3, 11]`)
- **The Cliff**: Bottom row between Start and Goal (States 37 to 46)
- **Actions**: Up ($\uparrow$), Right ($\rightarrow$), Down ($\downarrow$), Left ($\leftarrow$)

### Reward Structure
- **Step Penalty**: $-1$ reward for every move.
- **Cliff Falling**: $-100$ penalty and instant reset to the Start position.
- **Goal**: Terminal state.

---

## 🧠 SARSA Algorithm

**SARSA** is an **On-Policy Temporal Difference (TD) control** method. Unlike Q-Learning (which is off-policy and assumes optimal greedy actions for future steps), SARSA incorporates the actual action chosen by the current policy (including exploratory steps) into its value updates.

### Update Equation:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

Where:
- $\alpha$: Learning rate (default: `0.5`)
- $\gamma$: Discount factor (default: `0.99`)
- $\epsilon$: Exploration probability in $\epsilon$-greedy policy (default: `0.1`)

### Key Insight: SARSA vs Q-Learning
Because SARSA considers the probability of making a random exploratory step into the cliff, it learns a **safer route** that loops around the top of the grid. Q-Learning learns the mathematically optimal path right along the edge of the cliff, which suffers heavy penalties whenever an exploration action occurs.

---

## 📁 Repository Structure

```
cliff-problem-by-SARSA/
│
├── SARSA.ipynb         # Interactive Jupyter Notebook with step-by-step walkthrough
├── sarsa.py            # Modular SARSAAgent class implementation
├── main.py             # CLI script for training, evaluation, and visual plotting
├── requirements.txt    # Project dependencies
├── LICENSE             # MIT License
└── README.md           # Documentation
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/Vivritti/cliff-problem-by-SARSA.git
cd cliff-problem-by-SARSA
pip install -r requirements.txt
```

### 2. Run Training Script

Execute `main.py` to train the SARSA agent for 500 episodes and generate performance plots:

```bash
python main.py --episodes 500 --alpha 0.5 --gamma 0.99 --epsilon 0.1
```

### 3. Run Jupyter Notebook

Open and run `SARSA.ipynb`:

```bash
jupyter notebook SARSA.ipynb
```

---

## 📊 Hyperparameters

| Parameter | Description | Value |
| :--- | :--- | :--- |
| `alpha` ($\alpha$) | Learning Rate | `0.5` |
| `gamma` ($\gamma$) | Discount Factor | `0.99` |
| `epsilon` ($\epsilon$) | Exploration Rate | `0.1` |
| `episodes` | Total Training Episodes | `500` |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
