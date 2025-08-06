# 🚗 Active Inference – Mountain Car (Julia + RxInfer)

This project implements the classic **Mountain Car** problem using the **Active Inference** framework in Julia, powered by [`RxInfer.jl`](https://github.com/biaslab/RxInfer.jl). Instead of traditional reinforcement learning, the agent uses probabilistic reasoning and free energy minimization to solve the task.

---

## 📖 Problem Description

A group of friends is traveling to a camping site located on the biggest mountain in the Netherlands. Their electric car, low on battery, is stuck in a valley. The engine is too weak to drive straight up the hill.

To reach the camping site at the top, the car must first drive backward to build **momentum**, then accelerate forward — a task that involves planning and adapting to physical constraints like gravity and friction.

---

## 🤖 Active Inference Agent

We build an agent that:

1. **Models the physical world** (gravity, friction, engine control)
2. **Defines probabilistic beliefs** over states, actions, and goals
3. **Infers** the most probable actions to reach its target by minimizing **variational free energy**
4. **Adapts** its future expectations over time

---

## 🔍 Key Concepts

- **Generative model**: Describes how the agent *believes* the world works (physics, observations, goals)
- **Inference**: Updates beliefs based on new observations (posterior estimation)
- **Prediction**: Forecasts likely future states to inform actions
- **Free energy minimization**: Drives the agent’s planning and decision-making

