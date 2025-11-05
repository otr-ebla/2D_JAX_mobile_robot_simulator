# 2D JAX Mobile Robot Simulator

This repository provides a lightweight, fully vectorised 2D indoor simulator built with [JAX](https://github.com/google/jax). It can generate thousands of random indoor environments (rooms, corridors, doors) and simulate the motion of differential-drive robots and humans represented as discs. The simulator includes a fast 2D lidar ray-casting primitive and is designed for reinforcement learning workflows that rely on massive parallel rollouts accelerated by GPUs or TPUs.

## Features

- Procedural generation of indoor layouts using axis-aligned rooms with configurable door openings and optional corridors.
- Batched representation of wall segments that scales efficiently across large batches.
- Simple disc-based kinematics for robots and humans with configurable dynamics limits.
- Vectorised collision resolution against walls for robots and humans.
- GPU-friendly 2D lidar ray casting that returns range readings for each batched agent.
- Example script that demonstrates how to roll out the simulator across multiple environments in parallel.

## Installation

1. Install Python 3.10+ and create a virtual environment.
2. Install the JAX build that matches your accelerator before pulling in the rest of the dependencies:

   - **CUDA 12.x**

     ```bash
     pip install --upgrade "jax[cuda12_pip]==0.4.31" \
         -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
     ```

   - **CUDA 11.x**

     ```bash
     pip install --upgrade "jax[cuda11_pip]==0.4.31" \
         -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
     ```

   - **CPU only**

     ```bash
     pip install --upgrade "jax[cpu]==0.4.31"
     ```

   Installing one of the GPU wheels above ensures that `jaxlib` contains the CUDA
   runtime; otherwise JAX will fall back to CPU execution with a warning similar
   to the one shown in the issue description.

3. Clone this repository.
4. Install the remaining Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Pre-made files are also provided for GPU users who prefer a single command:

   ```bash
   # CUDA 12.x
   pip install -r requirements-cuda12.txt

   # CUDA 11.x
   pip install -r requirements-cuda11.txt
   ```

The simulator itself has no third-party dependencies beyond JAX and the
visualisation helpers.
1. Install Python 3.10+ along with JAX and its accelerator-specific build (see the [official JAX installation guide](https://github.com/google/jax#installation)).
2. Clone this repository.
3. (Optional) Create a virtual environment and install extra dependencies required for experiments.

The simulator itself has no third-party dependencies beyond JAX.

## Usage

A minimal end-to-end example is provided in [`examples/random_rollout.py`](examples/random_rollout.py). The snippet below shows the essential workflow:

```python
import jax
import jax.numpy as jnp

from jax_mobile_sim.environment import MapGenerationConfig, generate_map_batch
from jax_mobile_sim.simulator import SimulationConfig, SimulationState, RobotState, PeopleState, step_simulation, lidar_scan

key = jax.random.PRNGKey(0)
map_config = MapGenerationConfig()
sim_config = SimulationConfig()

maps = generate_map_batch(key, batch_size=1024, config=map_config)
state = SimulationState(
    robots=RobotState(
        position=jnp.zeros((1024, 1, 2)),
        velocity=jnp.zeros((1024, 1, 2)),
        heading=jnp.zeros((1024, 1)),
    ),
    people=PeopleState(position=jnp.zeros((1024, 5, 2)), velocity=jnp.zeros((1024, 5, 2))),
)

actions = jnp.zeros((1024, 1, 2))  # linear and angular commands
key, step_key = jax.random.split(key)
next_state = step_simulation(state, actions, maps, sim_config, step_key)
angles = jnp.linspace(-jnp.pi, jnp.pi, 180)
scan = lidar_scan(next_state.robots.position, angles, 10.0, maps)
```

The API is functional and can be wrapped inside `jax.vmap` or `jax.jit` to achieve high throughput for reinforcement learning training loops.

## Example rollout

To run the provided rollout example:

```bash
python examples/random_rollout.py
```

This script creates several environments, samples random actions, performs a few simulation steps, and prints robot positions along with lidar observation shapes. The same code path can be JIT-compiled for faster performance when running on GPUs/TPUs.

## Roadmap

Potential extensions include:

- Richer human behaviour models (goal-directed motion, social forces).
- Support for non-axis-aligned walls or mesh-based environments.
- Integration with RL libraries for on-policy/off-policy training loops.
- Visualisation utilities for debugging and dataset generation.

Contributions and suggestions are welcome!

