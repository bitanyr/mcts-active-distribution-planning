# Active Distribution Network (ADN) Planning using MCTS

## Project Overview

This repository presents an implementation of a coupled **physical-economic simulator** for the optimal placement of equipments in Active Distribution Networks (ADNs).

The ADN planning problem is formulated as a challenging **Mixed-Integer Nonlinear Programming (MINLP)** problem due to the nonlinear and non-convex characteristics of AC power flow equations and the presence of discrete investment decisions, such as selecting candidate buses for CB, SVC, DG and BESS installation.

To efficiently explore this complex decision space, this project employs **Monte Carlo Tree Search (MCTS)**, inspired by the decision-making framework of **AlphaZero**.

---

## Engineering Challenge

The increasing penetration of distributed energy resources is transforming conventional distribution networks into **active, decentralized, and economically dynamic systems**.

In this environment, maintaining acceptable voltage profiles and minimizing power losses are no longer the only planning objectives. The interaction between network operation and **energy market economics**, particularly under **Real-Time Pricing (RTP)**, introduces additional complexity into the planning problem.

Two major challenges arise:

### 1. Combinatorial Explosion

As the number of network nodes and available technologies increases, the number of possible technologies placement and sizing configurations grows rapidly.

For example, even in the standard **IEEE 33-bus distribution system**, considering multiple candidate locations, technology types, and sizing decisions can result in a large and highly complex search space.

### 2. Nonlinear Power System Behavior

Distribution networks typically have relatively high **R/X ratios**, making conventional transmission-oriented power flow formulations less suitable for radial distribution systems.

To provide a robust representation of the network physics, this project utilizes **DistFlow-based branch equations**, which are particularly suitable for radial distribution networks.

---

## Proposed Approach

The proposed framework combines **power system modeling, economic optimization, and intelligent search** into a unified planning environment.

### 1. Physics-Based Network Modeling

The network is modeled using **DistFlow branch equations** based on the formulation introduced by **Baran and Wu (1989)**.

The model captures key electrical quantities, including:

- Active and reactive power flows
- Voltage magnitude
- Active and reactive power losses
- Network operating constraints

This physics-based formulation allows the planning agent to evaluate candidate decisions directly within the electrical network model.

### 2. Physical-Economic Objective Function

Instead of optimizing only technical indicators such as power losses or voltage deviation, the proposed framework incorporates the **economic consequences of planning decisions**.

The objective function considers the network's overall cash flow, including:

- **CapEx (Capital Expenditure)**
  - installation costs

- **OpEx (Operational Expenditure)**
  - Energy purchased from the upstream grid
  - Fuel costs
  - Network power losses
  - Battery degradation costs
  - Other operational expenses

This enables the planning process to evaluate investment decisions from both **technical and economic perspectives**.

### 3. Intelligent Search Using MCTS

**Monte Carlo Tree Search (MCTS)** is used to navigate the large combinatorial decision space.

Instead of exhaustively evaluating every possible configuration, MCTS progressively explores promising planning actions based on their estimated long-term rewards.

The agent can therefore make decisions regarding:

- DG placement
- BESS placement
- CB placement
- SVC placement
- Sequential investment decisions

The resulting planning strategy considers not only immediate network performance, but also the **long-term economic impact** of infrastructure decisions.

---

## Overall Framework

The overall architecture can be summarized as:

**Planning Decision → MCTS → Network Simulation → Power Flow & Constraints → Economic Evaluation → Reward → Tree Update**

This creates a closed-loop interaction between the intelligent planning algorithm and the physics-based power system simulator.

---

## Key Features

- Active Distribution Network planning
- BESS placement
- Distributed Generation planning
- DistFlow-based radial network modeling
- Physical-economic objective function
- Real-Time Pricing (RTP) integration
- Battery degradation cost modeling
- Monte Carlo Tree Search (MCTS)
- IEEE 33-bus test system support

---

## Project Goal

The main goal of this project is to investigate how **intelligent search algorithms can be combined with physics-based power system models and economic objectives** to address complex planning problems in modern active distribution networks.

The framework is designed as a foundation for further research toward **AI-assisted distribution network planning, energy market integration, and intelligent infrastructure investment**.
