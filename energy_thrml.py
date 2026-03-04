import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def main():
    print("--- Energy Landscape Simulation (THRML / JAX Implementation) ---")
    
    # Define our goal: The system must naturally settle into this state.
    # We want 4 nodes, and the pattern: [+1, -1, +1, -1]
    # We will use the exact same logic as defined in energy.py
    
    nodes = [SpinNode() for _ in range(4)]
    
    # 1. Sculpting the landscape logically
    # Connect all pairs of nodes (fully connected graph)
    edges = [
        (nodes[0], nodes[1]), 
        (nodes[0], nodes[2]), 
        (nodes[0], nodes[3]),
        (nodes[1], nodes[2]),
        (nodes[1], nodes[3]),
        (nodes[2], nodes[3])
    ]
    
    # Target state: [+1, -1, +1, -1]
    # Based on energy.py:
    # Biases pull towards specific values
    strong_pull_factor = 1.0
    biases = jnp.array([1.0, -1.0, 1.0, -1.0]) * strong_pull_factor
    
    # Weights enforce alignment/opposition between connected nodes
    # For a fully connected 4-node graph, the ordering in our edges list is:
    # 0-1 (oppose -> -1)
    # 0-2 (align -> +1)
    # 0-3 (oppose -> -1)
    # 1-2 (oppose -> -1)
    # 1-3 (align -> +1)
    # 2-3 (oppose -> -1)
    strong_bond_factor = 2.0
    weights = jnp.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]) * strong_bond_factor
    
    # Temperature definition based on energy.py (1.0 or 0.5)
    # Inverse temperature (Beta) = 1.0 / Temperature
    temperature = 0.5
    beta = jnp.array(1.0 / temperature)
    
    print("Initializing the THRML Ising Energy-Based Model...")
    model = IsingEBM(nodes, edges, biases, weights, beta)
    
    # For a fully-connected graph, updating everything simultaneously is invalid in Block Gibbs.
    # We define 4 separate blocks, one for each node, to guarantee independent updates.
    free_blocks = [Block([nodes[0]]), Block([nodes[1]]), Block([nodes[2]]), Block([nodes[3]])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # Random key setup
    key = jax.random.key(123)
    k_init, k_samp = jax.random.split(key, 2)
    
    print("Setting up Hinton initialization (random starting state)...")
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    print(f"Goal (Desired State): [+1, -1, +1, -1]")
    print("Starting simulation with Block Gibbs Sampling...")
    
    # Define our schedule to match num_steps=10000 in energy.py 
    # steps_per_sample determines how many block sweeps happen before saving a state list.
    schedule = SamplingSchedule(n_warmup=100, n_samples=10000, steps_per_sample=1)
    
    # Sample state updates
    samples_tuple = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    # Extract the trajectory for the entire system (the first requested metric block)
    trajectory = samples_tuple[0]
    
    print("\n--- Results ---")
    
    # The trajectory will be of shape (n_samples, 4) containing float32 values of 1.0 or -1.0
    # Let's average the last 1000 samples to see the 'settled' equilibrium
    settled_avg = jnp.mean(trajectory[-1000:], axis=0)
    
    print(f"Target Expected: [ 1.0, -1.0,  1.0, -1.0]")
    print(f"Settled Avg:     {settled_avg}")
    
    # Final sample in the run
    final_sample = trajectory[-1]
    print(f"Final Settled:   {final_sample}")
    
    # Check if the final state matches
    target = jnp.array([1.0, -1.0, 1.0, -1.0])
    if jnp.allclose(final_sample, target):
        print("Success! The THRML system settled into the desired low-energy state.")
    else:
        print("Failed to stay in desired state.")

if __name__ == "__main__":
    main()
