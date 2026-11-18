# Traffic Signal Optimization with Reinforcement Learning
This repository implements four reinforcement-learning approaches for intelligent traffic signal control in SUMO:
- **Tabular Q-Learning**
- **Deep Q-Network (DQN)**
- **Double DQN**
- **Dueling DQN**

All models share the same simulation environment, training pipeline, and evaluation system, enabling a fair comparison.


# 1. Model Descriptions

## 1.1 Tabular Q-Learning
A discrete lookup table mapping `state → action value`.

**Update rule:**
```
Q(s,a) ← Q(s,a) + α [ r + γ max_a′ Q(s′,a′) − Q(s,a) ]
```

**Characteristics:**
- Simple but limited
- Requires discrete state representation
- Does not scale to visual or high-dimensional states


## 1.2 Deep Q-Network (DQN)
Uses a neural network to approximate `Q(s,a)`.

**Features:**
- Replay buffer  
- Target network  
- Epsilon-greedy exploration  


## 1.3 Double DQN
Fixes Q-value overestimation.

**Target calculation:**
```
a* = argmax(Q_online(s′))
target = r + γ * Q_target(s′, a*)
```


## 1.4 Dueling DQN
Splits the network into:
- Value stream `V(s)`
- Advantage stream `A(s,a)`

Final combination:
```
Q(s,a) = V(s) + (A(s,a) − mean(A(s,*)))
```

# 2. Training

Run:
```
python training_main.py
```


# 3. Testing

Run:
```
python testing_main.py
```

# 4. Metrics Logged

- Average waiting time  
- Queue length  
- Total delay  
- Episode reward  
- Vehicle throughput  
- Training time / simulation time  

All logged per episode.

---

# 5. Comparison of Models

| Model        | Stability | Bias | Performance | Scalability |
|--------------|-----------|------|-------------|-------------|
| Tabular Q    | Low       | High | Low         | Poor        |
| DQN          | Medium    | High | Medium      | Good        |
| Double DQN   | High      | Low  | High        | Good        |
| Dueling DQN  | Very High | Low  | Very High   | Excellent   |

---

# 6. Adding a New Model
Steps:
1. Create new class in `model.py`
2. Add selection logic in `training_main.py` + `testing_main.py`
3. Add config key under `[model]`
4. Define saving/loading logic
5. Add architecture in networks folder

---

# 7. Workflow Summary

```
State extraction → Agent action → SUMO step → Reward calculation → Replay memory → Neural update → Target update → Logging → Save model
```
