"""Example rollout showcasing the 2D JAX simulator."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

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
    current_robot_waypoints,
    *,
    env_index: int,
    render_state: dict,
    sim_config: SimulationConfig,
    frame_delay: float,
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
        render_state["config"] = render_config = RenderConfig(
            robot_radius=sim_config.robot_radius,
            person_radius=sim_config.person_radius,
        )
        render_state["renderer"] = renderer = render_environment
        render_state["plt"] = plt_module = plt

    if state.robots.position.shape[1] == 0:
        return render_state

    render_ax = renderer(
        state,
        maps,
        angles,
        lidar_distances,
        env_index=env_index,
        robot_index=0,
        robot_waypoints=current_robot_waypoints,
        config=render_config,
        ax=render_ax,
    )
    render_ax.figure.canvas.flush_events()
    plt_module.pause(max(frame_delay, 0.0))

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

    robot_state = RobotState(
        position=robot_pos,
        velocity=jnp.zeros_like(robot_pos),
        heading=jnp.zeros((map_batch_size, num_robots), dtype=robot_pos.dtype),
    )
    people_state = PeopleState(position=people_pos, velocity=jnp.zeros_like(people_pos))
    return SimulationState(robots=robot_state, people=people_state), maps


def generate_waypoints(
    key: jax.Array,
    map_batch_size: int,
    num_agents: int,
    num_waypoints: int,
    map_config: MapGenerationConfig,
    margin: float,
) -> np.ndarray:
    if num_agents == 0:
        return np.zeros((map_batch_size, num_agents, num_waypoints, 2), dtype=np.float32)

    waypoint_key_x, waypoint_key_y = jax.random.split(key)
    xs = jax.random.uniform(
        waypoint_key_x,
        (map_batch_size, num_agents, num_waypoints, 1),
        minval=margin,
        maxval=map_config.world_size[0] - margin,
    )
    ys = jax.random.uniform(
        waypoint_key_y,
        (map_batch_size, num_agents, num_waypoints, 1),
        minval=margin,
        maxval=map_config.world_size[1] - margin,
    )
    waypoints = jnp.concatenate([xs, ys], axis=-1)
    return np.asarray(waypoints)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run a random rollout in the 2D simulator.")
    parser.add_argument("--render", action="store_true", help="Enable matplotlib rendering for one environment.")
    parser.add_argument("--env-index", type=int, default=0, help="Environment index to render.")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps to run.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--num-people", type=int, default=5, help="People per environment.")
    parser.add_argument("--num-robots", type=int, default=1, help="Robots per environment.")
    parser.add_argument("--num-angles", type=int, default=360, help="LiDAR rays (uniform in [-pi, pi)).")
    parser.add_argument(
        "--lidar-max-range",
        type=float,
        default=30.0,
        help="LiDAR max range (values above 30 m are clipped).",
    )
    parser.add_argument(
        "--sim-speed",
        type=float,
        default=1.0,
        help="Simulation speed multiplier. 1.0 is real-time, larger values run faster.",
    )
    args = parser.parse_args(argv)

    if args.sim_speed <= 0.0:
        parser.error("--sim-speed must be positive.")

    # Configs
    base_sim_config = SimulationConfig()
    sim_config = replace(base_sim_config, dt=base_sim_config.dt * args.sim_speed)
    real_time_dt = base_sim_config.dt
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

    margin = 2.0 * max(sim_config.robot_radius, sim_config.person_radius)
    waypoint_key_robot, waypoint_key_people = jax.random.split(jax.random.PRNGKey(123))
    num_robot_waypoints = 5
    num_people_waypoints = 4
    robot_waypoints = generate_waypoints(
        waypoint_key_robot,
        args.batch_size,
        args.num_robots,
        num_robot_waypoints,
        map_config,
        margin,
    )
    people_waypoints = generate_waypoints(
        waypoint_key_people,
        args.batch_size,
        args.num_people,
        num_people_waypoints,
        map_config,
        margin,
    )
    robot_waypoint_indices = np.zeros((args.batch_size, args.num_robots), dtype=np.int32)
    people_waypoint_indices = np.zeros((args.batch_size, args.num_people), dtype=np.int32)

    # Controls + sensors
    actions_key = jax.random.PRNGKey(42)
    # LiDAR rays originate at the robot heading and progress clockwise
    angles = jnp.linspace(0.0, 2 * jnp.pi, args.num_angles, endpoint=False)

    # Rendering cache
    render_state = {"ax": None, "config": None, "renderer": None, "plt": None}
    env_to_render = int(jnp.clip(args.env_index, 0, args.batch_size - 1))

    lidar_max_range = min(args.lidar_max_range, 30.0)

    for step in range(args.steps):
        actions_key, sub = jax.random.split(actions_key)

        if args.num_robots > 0:
            current_targets = np.take_along_axis(
                robot_waypoints,
                robot_waypoint_indices[..., None, None],
                axis=2,
            ).squeeze(2)
            robot_positions = np.asarray(state.robots.position)
            robot_headings = np.asarray(state.robots.heading)
            deltas = current_targets - robot_positions
            distances = np.linalg.norm(deltas, axis=-1)
            reached = distances <= 0.3
            if np.any(reached):
                robot_waypoint_indices = (robot_waypoint_indices + reached.astype(np.int32)) % num_robot_waypoints
                current_targets = np.take_along_axis(
                    robot_waypoints,
                    robot_waypoint_indices[..., None, None],
                    axis=2,
                ).squeeze(2)
                deltas = current_targets - robot_positions
                distances = np.linalg.norm(deltas, axis=-1)

            norms = np.clip(distances[..., None], a_min=1e-6, a_max=None)
            directions = deltas / norms
            desired_heading = np.arctan2(directions[..., 1], directions[..., 0])
            heading_error = np.arctan2(
                np.sin(desired_heading - robot_headings), np.cos(desired_heading - robot_headings)
            )
            angular_speed = np.clip(
                heading_error / 0.5,
                -sim_config.max_robot_angular_speed,
                sim_config.max_robot_angular_speed,
            )
            robot_speed = 0.9 * sim_config.max_robot_speed
            linear_speed = robot_speed * np.clip(np.cos(heading_error), 0.0, 1.0)
            linear_speed = np.where(distances <= 0.1, 0.0, linear_speed)
            actions_np = np.stack([linear_speed, angular_speed], axis=-1).astype(np.float32)
            robot_speed = 0.9 * sim_config.max_robot_speed
            actions_np = (directions * robot_speed).astype(np.float32)
            actions = jnp.asarray(actions_np)
        else:
            current_targets = np.zeros((args.batch_size, args.num_robots, 2), dtype=np.float32)
            actions = jnp.zeros((args.batch_size, args.num_robots, 2), dtype=jnp.float32)

        if args.num_people > 0:
            people_targets = np.take_along_axis(
                people_waypoints,
                people_waypoint_indices[..., None, None],
                axis=2,
            ).squeeze(2)
            people_positions = np.asarray(state.people.position)
            people_deltas = people_targets - people_positions
            people_distances = np.linalg.norm(people_deltas, axis=-1)
            people_reached = people_distances <= 0.4
            if np.any(people_reached):
                people_waypoint_indices = (people_waypoint_indices + people_reached.astype(np.int32)) % num_people_waypoints
                people_targets = np.take_along_axis(
                    people_waypoints,
                    people_waypoint_indices[..., None, None],
                    axis=2,
                ).squeeze(2)
                people_deltas = people_targets - people_positions
                people_distances = np.linalg.norm(people_deltas, axis=-1)

            people_norms = np.clip(people_distances[..., None], a_min=1e-6, a_max=None)
            people_directions = people_deltas / people_norms
            desired_people_velocity = jnp.asarray(
                (people_directions * (0.8 * sim_config.max_person_speed)).astype(np.float32)
            )
            state = SimulationState(
                robots=state.robots,
                people=PeopleState(position=state.people.position, velocity=desired_people_velocity),
            )
        else:
            people_targets = np.zeros((args.batch_size, args.num_people, 2), dtype=np.float32)

        # Step simulation with higher-frequency dynamics
        step_key = jax.random.fold_in(sub, step)
        state = step_simulation(state, actions, maps, sim_config, step_key)

        # High-frequency LiDAR after the dynamics step
        # lidar_distances, _ = lidar_scan(
        #     state.robots.position,
        #     angles,
        #     lidar_max_range,
        #     maps,
        #     people_positions=state.people.position,
        #     person_radius=sim_config.person_radius,
        #     num_subsamples=sim_config.lidar_updates_per_step,
        #     origin_velocities=state.robots.velocity,
        #     people_velocities=state.people.velocity,
        #     dt=sim_config.dt,
        #     return_history=True,
        # )
        lidar_distances, _ = lidar_scan(
            state.robots.position,
            angles,
            lidar_max_range,
            maps,
            people_positions=state.people.position,
            person_radius=sim_config.person_radius,
            num_subsamples=sim_config.lidar_updates_per_step,
            origin_velocities=state.robots.velocity,
            people_velocities=state.people.velocity,
            dt=sim_config.dt,
            robot_headings=state.robots.heading,
            return_history=True,
        )

        # Minimal log
        if step % 10 == 0 or step == args.steps - 1:
            print(f"[step {step}] positions shape: {state.robots.position.shape} | lidar: {lidar_distances.shape}")

        # Optional render
        if args.render:
            frame_delay = real_time_dt / max(args.sim_speed, 1e-6)
            frame_delay = sim_config.dt / max(args.sim_speed * sim_config.dynamics_substeps, 1e-6)
            render_state = _maybe_render(
                state,
                maps,
                angles,
                lidar_distances,
                current_targets,
                env_index=env_to_render,
                render_state=render_state,
                sim_config=sim_config,
                frame_delay=frame_delay,
            )

    # keep window open briefly if rendering
    if args.render and render_state.get("plt") is not None:
        render_state["plt"].pause(0.5)


if __name__ == "__main__":
    main()
