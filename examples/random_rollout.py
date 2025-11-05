"""Example rollout showcasing the 2D JAX simulator."""
from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp

# Make repo importable when running from examples/
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


def _maybe_render(
    state: SimulationState,
    maps: IndoorMapBatch,
    angles: jax.Array,
    lidar_distances: jax.Array,
    *,
    env_index: int,
    render_state: dict,
) -> dict:
    """Render the requested environment on demand (lazy init)."""
    render_ax = render_state.get("ax")
    render_config = render_state.get("config")
    renderer = render_state.get("renderer")
    plt_module = render_state.get("plt")

    if renderer is None:
        from matplotlib import pyplot as plt
        from jax_mobile_sim.rendering import RenderConfig, render_environment

        plt.ion()
        render_state["config"] = render_config = RenderConfig()
        render_state["renderer"] = renderer = render_environment
        render_state["plt"] = plt_module = plt

    render_ax = renderer(
        state,
        maps,
        angles,
        lidar_distances,
        env_index=env_index,
        robot_index=0,
        config=render_config,
        ax=render_ax,
    )
    render_ax.figure.canvas.flush_events()
    plt_module.pause(0.001)

    render_state["ax"] = render_ax
    return render_state


def initialize_state(
    key: jax.Array,
    map_batch_size: int,
    num_robots: int,
    num_people: int,
    sim_config: SimulationConfig,
    map_config: MapGenerationConfig,
) -> tuple[SimulationState, IndoorMapBatch]:
    """Create maps + random initial positions for robots and people."""
    map_key, robot_key, people_key = jax.random.split(key, 3)
    maps = generate_map_batch(map_key, map_batch_size, map_config)
    margin = 2.0 * max(sim_config.robot_radius, sim_config.person_radius)

    def sample_positions(rng, count):
        xs = jax.random.uniform(
            rng, (map_batch_size, count, 1),
            minval=margin, maxval=map_config.world_size[0] - margin
        )
        ys = jax.random.uniform(
            jax.random.fold_in(rng, 1), (map_batch_size, count, 1),
            minval=margin, maxval=map_config.world_size[1] - margin
        )
        return jnp.concatenate([xs, ys], axis=-1)

    robot_pos = sample_positions(robot_key, num_robots)
    people_pos = sample_positions(people_key, num_people)

    robot_state = RobotState(position=robot_pos, velocity=jnp.zeros_like(robot_pos))
    people_state = PeopleState(position=people_pos, velocity=jnp.zeros_like(people_pos))
    return SimulationState(robots=robot_state, people=people_state), maps


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run a random rollout in the 2D simulator.")
    parser.add_argument("--render", action="store_true", help="Enable matplotlib rendering for one environment.")
    parser.add_argument("--env-index", type=int, default=0, help="Environment index to render.")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps to run.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--num-people", type=int, default=5, help="People per environment.")
    parser.add_argument("--num-robots", type=int, default=1, help="Robots per environment.")
    parser.add_argument("--num-angles", type=int, default=360, help="LiDAR rays (uniform in [-pi, pi)).")
    parser.add_argument("--lidar-max-range", type=float, default=10.0, help="LiDAR max range.")
    args = parser.parse_args(argv)

    # Configs
    sim_config = SimulationConfig()
    map_config = MapGenerationConfig()

    # Init state
    key = jax.random.PRNGKey(0)
    state, maps = initialize_state(
        key,
        args.batch_size,
        args.num_robots,
        args.num_people,
        sim_config,
        map_config,
    )

    # Controls + sensors
    actions_key = jax.random.PRNGKey(42)
    angles = jnp.linspace(-jnp.pi, jnp.pi, args.num_angles, endpoint=False)

    # Rendering cache
    render_state = {"ax": None, "config": None, "renderer": None, "plt": None}
    env_to_render = int(jnp.clip(args.env_index, 0, args.batch_size - 1))

    for step in range(args.steps):
        # Random actions in [-1, 1]
        actions_key, sub = jax.random.split(actions_key)
        actions = jax.random.uniform(
            sub,
            (args.batch_size, args.num_robots, 2),
            minval=-1.0, maxval=1.0,
        )

        # LiDAR (before stepping, for visualization of current state)
        lidar_distances = lidar_scan(state.robots.position, angles, args.lidar_max_range, maps)

        # Step simulation
        state = step_simulation(state, actions, maps, sim_config, jax.random.fold_in(sub, step))

        # Minimal log
        if step % 10 == 0 or step == args.steps - 1:
            print(f"[step {step}] positions shape: {state.robots.position.shape} | lidar: {lidar_distances.shape}")

        # Optional render
        if args.render:
            render_state = _maybe_render(
                state,
                maps,
                angles,
                lidar_distances,
                env_index=env_to_render,
                render_state=render_state,
            )

    # keep window open briefly if rendering
    if args.render and render_state.get("plt") is not None:
        render_state["plt"].pause(0.5)


if __name__ == "__main__":
    main()
