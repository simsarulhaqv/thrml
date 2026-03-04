import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def main():
    print("Setting up a small Ising chain with two-color block Gibbs sampling...")
    
    # Create 5 spin nodes
    nodes = [SpinNode() for _ in range(5)]
    
    # Create edges between adjacent nodes (chain topology)
    edges = [(nodes[i], nodes[i+1]) for i in range(4)]
    
    # Set biases and weights
    biases = jnp.zeros((5,))
    weights = jnp.ones((4,)) * 0.5
    beta = jnp.array(1.0)
    
    print("Initializing Ising Energy-Based Model...")
    model = IsingEBM(nodes, edges, biases, weights, beta)
    
    # Define free blocks for sampling (two-color scheme)
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # PRNG setup
    key = jax.random.key(0)
    k_init, k_samp = jax.random.split(key, 2)
    
    print("Running hinton initialization...")
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    # Define sampling schedule (10k samples)
    schedule = SamplingSchedule(n_warmup=100, n_samples=10000, steps_per_sample=2)
    
    print("Starting sample collection...")
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    print("Sampling complete!")
    print(f"Sample shapes (list of arrays): {[s.shape for s in samples]}")

if __name__ == "__main__":
    main()
