"""Optimized rollout for high-frequency 2D simulator."""
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

DT = 0.1

from jax_mobile_sim.environment import IndoorMapBatch, MapGenerationConfig, generate_map_batch
from jax_mobile_sim.simulator import (
    PeopleState,
    RobotState,
    SimulationConfig,
    SimulationState,
    lidar_scan,
    step_simulation,
)

def _maybe_render_fast(
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
    """Fast rendering with minimal overhead."""
    render_ax = render_state.get("ax")
    render_config = render_state.get("config")
    renderer = render_state.get("renderer")
    plt_module = render_state.get("plt")

    if renderer is None:
        from matplotlib import pyplot as plt
        from jax_mobile_sim.rendering import RenderConfig, render_environment

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        render_state["config"] = render_config = RenderConfig(
            robot_radius=sim_config.robot_radius,
            person_radius=sim_config.person_radius,
            lidar_alpha=0.2,  # Reduced alpha for faster rendering
        )
        render_state["renderer"] = renderer = render_environment
        render_state["plt"] = plt_module = plt
        render_state["ax"] = ax
        render_state["fig"] = fig

    if state.robots.position.shape[1] == 0:
        return render_state

    # Use the existing axis for faster updates
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
    
    # Minimal canvas update
    render_ax.figure.canvas.draw()
    render_ax.figure.canvas.flush_events()
    
    # Very short pause for responsiveness
    plt_module.pause(max(frame_delay * 0.1, 0.001))  # Reduced pause

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
) -> jnp.ndarray:
    if num_agents == 0:
        return jnp.zeros((map_batch_size, num_agents, num_waypoints, 2), dtype=jnp.float32)

    @jax.jit
    def generate_waypoints_jit(key):
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
        return jnp.concatenate([xs, ys], axis=-1)

    return generate_waypoints_jit(key)

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run optimized random rollout in the 2D simulator.")
    parser.add_argument("--render", action="store_true", help="Enable matplotlib rendering for one environment.")
    parser.add_argument("--env-index", type=int, default=0, help="Environment index to render.")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of parallel environments.")  # Reduced
    parser.add_argument("--num-people", type=int, default=3, help="People per environment.")  # Reduced
    parser.add_argument("--num-robots", type=int, default=1, help="Robots per environment.")
    parser.add_argument("--num-angles", type=int, default=90, help="LiDAR rays (reduced for performance).")  # Reduced
    parser.add_argument("--lidar-max-range", type=float, default=15.0, help="LiDAR max range.")  # Reduced
    parser.add_argument("--sim-speed", type=float, default=2.0, help="Simulation speed multiplier.")  # Increased
    args = parser.parse_args(argv)

    # OPTIMIZED Configs - Reduced complexity
    base_sim_config = SimulationConfig()
    sim_config = replace(
        base_sim_config, 
        dt=DT,  # Smaller dt for stability at high speed
        dynamics_substeps=1,  # Reduced from 4
        lidar_updates_per_step=1,  # Reduced from 12
        max_robot_speed=2.0,  # Increased for faster movement
        max_person_speed=1.5
    )
    
    map_config = MapGenerationConfig(max_segments=80)  # Reduced map complexity

    # Init state - DON'T JIT compile map generation
    key = jax.random.PRNGKey(0)
    state, maps = initialize_state(
        key,
        args.batch_size,
        args.num_robots,
        args.num_people,
        sim_config,
        map_config,
    )

    # Waypoints
    margin = 2.0 * max(sim_config.robot_radius, sim_config.person_radius)
    waypoint_key_robot, waypoint_key_people = jax.random.split(jax.random.PRNGKey(123))
    
    num_robot_waypoints = 3  # Reduced
    num_people_waypoints = 2  # Reduced
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
    
    robot_waypoint_indices = jnp.zeros((args.batch_size, args.num_robots), dtype=jnp.int32)
    people_waypoint_indices = jnp.zeros((args.batch_size, args.num_people), dtype=jnp.int32)

    # Controls + sensors
    actions_key = jax.random.PRNGKey(42)
    angles = jnp.linspace(0.0, 2 * jnp.pi, args.num_angles, endpoint=False)

    # JIT compiled simulation step
    @jax.jit
    def simulation_step(state, actions, maps, key):
        return step_simulation(state, actions, maps, sim_config, key)

    # JIT compiled lidar
    @jax.jit
    def lidar_step(state, maps):
        return lidar_scan(
            state.robots.position,
            angles,
            args.lidar_max_range,
            maps,
            people_positions=state.people.position,
            person_radius=sim_config.person_radius,
            num_subsamples=1,  # Reduced
            origin_velocities=state.robots.velocity,
            people_velocities=state.people.velocity,
            robot_headings=state.robots.heading,
            dt=sim_config.dt,
            return_history=False,  # No history for performance
        )

    # Rendering cache
    render_state = {"ax": None, "config": None, "renderer": None, "plt": None}
    env_to_render = int(jnp.clip(args.env_index, 0, args.batch_size - 1))

    print(f"Starting optimized simulation with {args.steps} steps...")
    print(f"Target FPS: {1.0/sim_config.dt:.1f}")

    import time
    total_start = time.perf_counter()
    
    for step in range(args.steps):
        step_start = time.perf_counter()
        actions_key, sub = jax.random.split(actions_key)

        # Simplified control logic
        if args.num_robots > 0:
            # Convert to numpy only for control logic
            current_targets = np.take_along_axis(
                np.array(robot_waypoints),
                np.array(robot_waypoint_indices)[..., None, None],
                axis=2,
            ).squeeze(2)
            robot_positions = np.array(state.robots.position)
            robot_headings = np.array(state.robots.heading)
            
            deltas = current_targets - robot_positions
            distances = np.linalg.norm(deltas, axis=-1)
            reached = distances <= 0.5  # Increased threshold
            
            if np.any(reached):
                robot_waypoint_indices = (robot_waypoint_indices + reached.astype(np.int32)) % num_robot_waypoints
                current_targets = np.take_along_axis(
                    np.array(robot_waypoints),
                    np.array(robot_waypoint_indices)[..., None, None],
                    axis=2,
                ).squeeze(2)
                deltas = current_targets - robot_positions
            
            # Simple control: move toward target
            actions_np = deltas * 0.5  # Reduced gain
            actions = jnp.asarray(actions_np)
        else:
            current_targets = jnp.zeros((args.batch_size, args.num_robots, 2), dtype=jnp.float32)
            actions = jnp.zeros((args.batch_size, args.num_robots, 2), dtype=jnp.float32)

        # Step simulation
        step_key = jax.random.fold_in(sub, step)
        state = simulation_step(state, actions, maps, step_key)

        # Lidar - only every few steps for performance
        
        lidar_distances = lidar_step(state, maps)
        

        step_time = time.perf_counter() - step_start
        
        # Logging - reduced frequency
        if step % 20 == 0 or step == args.steps - 1:
            actual_fps = 1.0 / step_time if step_time > 0 else 0
            print(f"[step {step:3d}] time: {step_time*1000:.1f}ms, FPS: {actual_fps:.1f}")

        # Optional render - reduced frequency
        if args.render:  # Render every 3 steps
            frame_delay = sim_config.dt / max(args.sim_speed, 1e-6)
            render_state = _maybe_render_fast(
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

    total_time = time.perf_counter() - total_start
    avg_fps = args.steps / total_time
    print(f"\n=== Simulation Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Target vs Actual: {1.0/sim_config.dt:.1f} vs {avg_fps:.1f}")

    if args.render and render_state.get("plt") is not None:
        render_state["plt"].pause(1.0)

if __name__ == "__main__":
    main()