"""Example rollout showcasing the 2D JAX simulator."""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jax_mobile_sim.environment import IndoorMapBatch, MapGenerationConfig, generate_map_batch
from jax_mobile_sim.simulator import (
    PeopleState,
    RobotState,
    SimulationConfig,
    SimulationState,
    lidar_scan,
    step_simulation,
)


def initialize_state(
    key: jax.Array,
    map_batch_size: int,
    num_robots: int,
    num_people: int,
    sim_config: SimulationConfig,
    map_config: MapGenerationConfig,
) -> tuple[SimulationState, IndoorMapBatch]:
    map_key, robot_key, people_key = jax.random.split(key, 3)
    maps = generate_map_batch(map_key, map_batch_size, map_config)
    margin = 2.0 * max(sim_config.robot_radius, sim_config.person_radius)

    def sample_positions(rng, count):
        xs = jax.random.uniform(rng, (map_batch_size, count, 1), minval=margin, maxval=map_config.world_size[0] - margin)
        ys = jax.random.uniform(
            jax.random.fold_in(rng, 1),
            (map_batch_size, count, 1),
            minval=margin,
            maxval=map_config.world_size[1] - margin,
        )
        return jnp.concatenate([xs, ys], axis=-1)

    robot_pos = sample_positions(robot_key, num_robots)
    people_pos = sample_positions(people_key, num_people)

    robot_state = RobotState(position=robot_pos, velocity=jnp.zeros_like(robot_pos))
    people_state = PeopleState(position=people_pos, velocity=jnp.zeros_like(people_pos))
    return SimulationState(robots=robot_state, people=people_state), maps


def main():
    batch_size = 8
    num_robots = 1
    num_people = 5
    sim_config = SimulationConfig()
    map_config = MapGenerationConfig()

    key = jax.random.PRNGKey(0)
    state, maps = initialize_state(key, batch_size, num_robots, num_people, sim_config, map_config)

    actions_key = jax.random.PRNGKey(42)
    angles = jnp.linspace(-jnp.pi, jnp.pi, 180)

    for step in range(5):
        actions_key, sub = jax.random.split(actions_key)
        actions = jax.random.uniform(sub, (batch_size, num_robots, 2), minval=-1.0, maxval=1.0)
        lidar_distances = lidar_scan(state.robots.position, angles, 10.0, maps)
        state = step_simulation(state, actions, maps, sim_config, jax.random.fold_in(sub, step))
        print(f"Step {step}: robot positions\n{state.robots.position}")
        print(f"Lidar distances shape: {lidar_distances.shape}")


if __name__ == "__main__":
    main()

