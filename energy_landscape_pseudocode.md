# Energy Landscape: Logic & Pseudocode

This document outlines the core logic of an energy landscape model, abstracting away technical implementation details (like JAX, PyTrees, or hardware acceleration constraints).

## Core Concept
In an energy landscape, every possible state of a system is assigned an "Energy" value. 
The system naturally evolves over time, probabilistically moving towards states with lower energy. 
The global minimum is the "desired output."

## 1. Define the System (The Nodes)
- Create a set of components (nodes) that can be in different states.
- For a binary system (like an Ising model), a node's state is either `+1` (Up) or `-1` (Down).
- Let $X = [x_1, x_2, ..., x_N]$ be the current state of the system.

## 2. Define the Energy Function $E(X)$
The energy of any state $X$ is defined by interactions between nodes and external biases.
- **Biases ($B_i$)**: Does node $i$ prefer to be $+1$ or $-1$? 
  - (A positive bias towards $+1$ Lowers the energy if $x_i = +1$).
- **Weights ($W_{ij}$)**: Do nodes $i$ and $j$ prefer to have the same state or opposite states?
  - (A positive weight Lowers the energy if $x_i$ and $x_j$ are aligned).

```text
Function Calculate_Energy(State X):
    total_energy = 0
    
    // 1. Add up the pairwise interactions (Edges)
    For every pair of connected nodes (i, j):
        interaction = Weights[i][j] * X[i] * X[j]
        total_energy = total_energy - interaction  // Subtract because we want *lower* energy for preferred states
        
    // 2. Add up the individual node biases
    For every node i in the system:
        node_bias = Biases[i] * X[i]
        total_energy = total_energy - node_bias
        
    Return total_energy
```

## 3. Designing the Landscape (Setting Weights & Biases)
To make the system "settle" into a specific desired output, we sculpt the landscape so that the desired state is exactly at the bottom of the deepest valley.

```text
Function Sculpt_Landscape(Desired_State D):
    // D is our target, e.g., [+1, -1, +1, -1]
    
    // Set biases to pull nodes toward their desired value
    For each node i:
        Biases[i] = D[i] * strong_pull_factor
        
    // Set weights to enforce the relationships in the desired state
    For each pair of connected nodes (i, j):
        If D[i] == D[j]:
            Weights[i][j] = +1.0 * strong_bond_factor  // They should align
        Else:
            Weights[i][j] = -1.0 * strong_bond_factor  // They should oppose
            
    Return Biases, Weights
```

## 4. The Settling Process (Gibbs Sampling)
Instead of exhaustively checking every possible state (which is impossible for large systems), the system explores the landscape. It takes steps, usually accepting changes that lower the energy, but occasionally accepting higher-energy changes (due to "temperature" / thermal fluctuations) to escape local, shallow valleys.

```text
Function Simulate_Settling(Nodes, Biases, Weights, Temperature, Num_Steps):
    
    // 1. Start somewhere random
    Current_State = Randomize_All_Nodes()
    
    // 2. Explore
    Loop for step from 1 to Num_Steps:
    
        // Pick a node to consider flipping
        Choose a random node i
        
        // Calculate what the system's energy would be if x_i was +1 vs -1
        Energy_if_Plus1 = Calculate_Energy(Current_State with x_i = +1)
        Energy_if_Minus1 = Calculate_Energy(Current_State with x_i = -1)
        
        // Convert the energy difference into a probability using Physics (Boltzmann distribution)
        // Lower energy states have exponentially higher probability.
        // Higher Temperature makes the probabilities flatter (more random jumping).
        Prob_Plus1 = exp(-Energy_if_Plus1 / Temperature)
        Prob_Minus1 = exp(-Energy_if_Minus1 / Temperature)
        
        Normalize probabilities so they sum to 1.0
        
        // Flip a weighted coin to decide the node's new state
        Current_State[i] = Random_Choice(Options=[+1, -1], Probabilities=[Prob_Plus1, Prob_Minus1])
        
    // After many steps, the system will spend most of its time in the lowest energy valley
    Return Current_State
```

## Summary
By carefully choosing `Weights` and `Biases` mapping to a problem, the universe's natural tendency to minimize energy (simulated via sampling) automatically computations the solution to that problem for you.
