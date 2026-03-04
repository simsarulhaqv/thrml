import random
import math

# Energy Landscape Model: Simple Python Implementation
# This script implements the pseudocode from energy_landscape_pseudocode.md
# It does not use THRML or JAX. It uses pure Python to demonstrate the logic.

# In our problem, state values are either +1 or -1
STATES = [1, -1]

def calculate_energy(state, biases, weights):
    """
    Calculates the total energy of a given state.
    Lower energy represents a more "preferred" state.
    """
    total_energy = 0.0
    num_nodes = len(state)
    
    # 1. Add up the pairwise interactions (Edges)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # Only check each pair once
            interaction = weights[i][j] * state[i] * state[j]
            total_energy -= interaction
            
    # 2. Add up the individual node biases
    for i in range(num_nodes):
        node_bias = biases[i] * state[i]
        total_energy -= node_bias
        
    return total_energy

def sculpt_landscape(desired_state):
    """
    Sets the weights and biases such that the desired_state 
    has the lowest possible energy (global minimum).
    """
    num_nodes = len(desired_state)
    
    # Initialize biases and weights
    biases = [0.0] * num_nodes
    weights = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    strong_pull_factor = 1.0
    strong_bond_factor = 2.0
    
    # Set biases to pull nodes toward their desired value
    for i in range(num_nodes):
        biases[i] = desired_state[i] * strong_pull_factor
        
    # Set weights to enforce the relationships in the desired state
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if desired_state[i] == desired_state[j]:
                # They should align to lower energy (weight is positive)
                weights[i][j] = 1.0 * strong_bond_factor
                weights[j][i] = 1.0 * strong_bond_factor 
            else:
                # They should oppose each other to lower energy (weight is negative)
                weights[i][j] = -1.0 * strong_bond_factor
                weights[j][i] = -1.0 * strong_bond_factor
                
    return biases, weights

def simulate_settling(biases, weights, temperature, num_steps):
    """
    Uses Gibbs sampling to explore the landscape.
    """
    num_nodes = len(biases)
    
    # 1. Start somewhere random
    current_state = [random.choice(STATES) for _ in range(num_nodes)]
    print(f"Initial Random State: {current_state}")
    
    # 2. Explore
    for step in range(num_steps):
        # Pick a node to consider flipping
        i = random.randint(0, num_nodes - 1)
        
        # Calculate what the system's energy would be if x_i was +1 vs -1
        # To do this, we create copies of the state
        state_if_plus1 = list(current_state)
        state_if_plus1[i] = 1
        
        state_if_minus1 = list(current_state)
        state_if_minus1[i] = -1
        
        energy_plus1 = calculate_energy(state_if_plus1, biases, weights)
        energy_minus1 = calculate_energy(state_if_minus1, biases, weights)
        
        # Convert the energy difference into probabilities (Boltzmann distribution)
        # We subtract the maximum to improve numerical stability of math.exp
        max_energy = max(-energy_plus1 / temperature, -energy_minus1 / temperature)
        weight_plus1 = math.exp(-energy_plus1 / temperature - max_energy)
        weight_minus1 = math.exp(-energy_minus1 / temperature - max_energy)
        
        # Normalize
        total_weight = weight_plus1 + weight_minus1
        prob_plus1 = weight_plus1 / total_weight
        
        # Flip a weighted coin to decide the node's new state
        if random.random() < prob_plus1:
            current_state[i] = 1
        else:
            current_state[i] = -1
            
        # Optional: Print progress occasionally
        if (step + 1) % 500 == 0:
            current_energy = calculate_energy(current_state, biases, weights)
            # print(f"Step {step + 1:4d} | State: {current_state} | Energy: {current_energy}")

    return current_state

def main():
    print("--- Energy Landscape Simulation (Pure Python) ---")
    
    # Define our goal: The system must naturally settle into this state.
    # Let's say we have 4 nodes, and we want the pattern: [+1, -1, +1, -1]
    desired_state = [1, -1, 1, -1]
    print(f"Goal (Desired State): {desired_state}")
    
    # 1. Sculpt the landscape
    biases, weights = sculpt_landscape(desired_state)
    
    target_energy = calculate_energy(desired_state, biases, weights)
    print(f"Energy of Desired State: {target_energy} (Global Minimum)")
    
    # 2. Simulate the process of settling into the valley
    # A higher temperature allows exploring. We want it to lower over time (simulated annealing) 
    # but for a simple Gibbs sample we can just use a moderate temperature.
    # We increase the number of steps and define a slightly lower temperature to make the valley "steeper"
    temperature = 0.5 
    num_steps = 10000
    
    print("\nStarting simulation...")
    final_state = simulate_settling(biases, weights, temperature, num_steps)
    
    final_energy = calculate_energy(final_state, biases, weights)
    
    print("\n--- Results ---")
    print(f"Target Expected: {desired_state}")
    print(f"Final Settled:   {final_state}")
    print(f"Final Energy:    {final_energy}")
    
    if final_state == desired_state:
        print("Success! The system settled into the desired low-energy state.")
    else:
        print("Failed to reach desired state. Try increasing num_steps or tuning temperature.")

if __name__ == "__main__":
    main()
