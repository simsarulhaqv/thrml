import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def main():
    print("Simulating an Energy Landscape with THRML")
    print("-----------------------------------------")
    print("Goal: The system naturally 'settles' into a low-energy state.")
    
    # We'll create a simple 4-node fully connected graph (a small spin glass)
    # The energy function we'll define is: E(x) = -sum(weight_ij * x_i * x_j) - sum(bias_i * x_i)
    # We want a specific state to be the global minimum (lowest energy).
    # Let's say we want the desired state to be: [+1, -1, +1, -1]
    
    nodes = [SpinNode() for _ in range(4)]
    
    # Fully connected edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edges = [
        (nodes[0], nodes[1]), 
        (nodes[0], nodes[2]), 
        (nodes[0], nodes[3]),
        (nodes[1], nodes[2]),
        (nodes[1], nodes[3]),
        (nodes[2], nodes[3])
    ]
    
    # We set biases and weights such that the state [+1, -1, +1, -1] has lowest energy.
    # A simple way to do this is to set weights = +1 for aligned desired spins, and -1 for anti-aligned.
    # Desired = [1, -1, 1, -1]
    # Edge (0,1): 1 * -1 = -1 -> weight = -1
    # Edge (0,2): 1 * 1 = +1 -> weight = +1
    # Edge (0,3): 1 * -1 = -1 -> weight = -1
    # Edge (1,2): -1 * 1 = -1 -> weight = -1
    # Edge (1,3): -1 * -1 = +1 -> weight = +1
    # Edge (2,3): 1 * -1 = -1 -> weight = -1
    
    # We want strong weights to create a deep energy valley
    weights = jnp.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]) * 2.0
    
    # Biases push the nodes towards their desired values
    biases = jnp.array([1.0, -1.0, 1.0, -1.0]) * 1.0
    
    beta = jnp.array(1.0) # Inverse temperature
    
    print("Initializing the Energy-Based Model...")
    model = IsingEBM(nodes, edges, biases, weights, beta)
    
    # Block Gibbs sampling needs independent blocks.
    # For a fully-connected graph, each node must be its own block if we are strictly updating independently, 
    # but we can group them if needed. Here we just use two arbitrary blocks.
    # Wait, in a fully connected graph, all nodes are neighbors. 
    # To use valid Block Gibbs sampling, nodes in the same block must NOT share an edge.
    # So we must sample them one by one (4 blocks) for exact Gibbs, or use a schedule.
    # Let's use 4 blocks:
    free_blocks = [Block([nodes[0]]), Block([nodes[1]]), Block([nodes[2]]), Block([nodes[3]])]
    
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    
    print("Setting up Hinton initialization (random starting state)...")
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    print(f"Sampling landscape. The system should settle into the desired state: [1, -1, 1, -1]")
    # We run for a decent amount of samples to see if it settles.
    schedule = SamplingSchedule(n_warmup=100, n_samples=5000, steps_per_sample=2)
    
    samples_tuple = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    # the samples for the Block(nodes) is returned
    trajectory = samples_tuple[0]
    
    print("Sampling complete. Analyzing the settled state...")
    # Calculate the average over the last 1000 samples to see the 'settled' state
    settled_avg = jnp.mean(trajectory[-1000:], axis=0)
    
    print(f"Expected State: [ 1.0, -1.0,  1.0, -1.0]")
    print(f"Settled Avg:   {settled_avg}")
    
    # Check if the last sample is the desired state
    last_sample = trajectory[-1]
    print(f"Last Sample:   {last_sample}")

if __name__ == "__main__":
    main()
