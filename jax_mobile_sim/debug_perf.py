import time, functools, os, sys

# --- Robust imports whether run as module or script ---
try:
    from jax_mobile_sim.simulator import (
        SimulationConfig, SimulationState, RobotState, PeopleState,
        step_simulation, lidar_scan,
    )
    from jax_mobile_sim.environment import generate_map_batch, MapGenerationConfig
except ImportError:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from jax_mobile_sim.simulator import (
        SimulationConfig, SimulationState, RobotState, PeopleState,
        step_simulation, lidar_scan,
    )
    from jax_mobile_sim.environment import generate_map_batch, MapGenerationConfig

import jax
import jax.numpy as jnp

def _block(x):
    jax.tree_util.tree_map(lambda a: getattr(a, "block_until_ready", lambda: None)(), x)
    return x

def time_call(name, f, *args, warmup=3, iters=20, **kwargs):
    # Warm-up (JIT compile)
    for _ in range(warmup):
        _block(f(*args, **kwargs))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(*args, **kwargs)
        _block(out)
    dt = (time.perf_counter() - t0) / iters
    print(f"[T] {name:<26} {dt*1000:7.2f} ms  |  {1.0/dt:8.1f} FPS")
    return out

def main():
    key = jax.random.key(0)

    # ---- Build optimized test batch ----
    B, R, P, A = 2, 4, 8, 180  # Reduced sizes for performance
    maps = generate_map_batch(key, B, MapGenerationConfig(max_segments=80))  # Reduced segments
    key, k1, k2, k3 = jax.random.split(key, 4)

    robot_pos = jax.random.uniform(k1, (B, R, 2), minval=1.0, maxval=5.0)
    robot_head = jax.random.uniform(k2, (B, R), minval=-jnp.pi, maxval=jnp.pi)
    robot_vel = jnp.zeros((B, R, 2))

    people_pos = jax.random.uniform(k3, (B, P, 2), minval=1.0, maxval=5.0)
    people_vel = jnp.zeros_like(people_pos)

    state = SimulationState(
        robots=RobotState(position=robot_pos, velocity=robot_vel, heading=robot_head),
        people=PeopleState(position=people_pos, velocity=people_vel),
    )

    # Optimized config for performance
    cfg = SimulationConfig(
        dt=0.05, 
        dynamics_substeps=1,  # Reduced from 4
        lidar_updates_per_step=1,  # Reduced from 12
        max_robot_speed=2.0,  # Increased for faster movement
        max_person_speed=1.5
    )
    
    linear = jnp.full((B, R), 0.8)
    angular = jnp.full((B, R), 0.1)
    actions = jnp.stack([linear, angular], axis=-1)
    
    # Define angles for lidar - reduced resolution
    angles = jnp.linspace(0, 2 * jnp.pi, A, endpoint=False)

    # ---- JIT wrappers ----
    step_sim_jit = jax.jit(
        lambda s, a, m, key: step_simulation(s, a, m, cfg, key)
    )

    lidar_jit = jax.jit(
        functools.partial(lidar_scan, num_subsamples=1, return_history=False),
        static_argnames=("max_range", "num_subsamples", "return_history"),
    )

    # ---- Time step_simulation ----
    print("=== Performance Benchmark ===")
    state = time_call(
        "step_simulation()",
        step_sim_jit,
        state, actions, maps,
        key=key
    )

    # ---- Time lidar_scan ----
    dists = time_call(
        "lidar_scan()",
        lidar_jit,
        state.robots.position, angles, 20.0, maps,  # Reduced max_range
        people_positions=state.people.position,
        person_radius=cfg.person_radius,
        origin_velocities=state.robots.velocity,
        people_velocities=state.people.velocity,
        robot_headings=state.robots.heading,
        dt=cfg.dt,
    )

    # ---- Combined pass ----
    def step_then_lidar(s, k):
        s2 = step_sim_jit(s, actions, maps, k)
        d = lidar_jit(
            s2.robots.position, angles, 20.0, maps,
            people_positions=s2.people.position,
            person_radius=cfg.person_radius,
            origin_velocities=s2.robots.velocity,
            people_velocities=s2.people.velocity,
            robot_headings=s2.robots.heading,
            dt=cfg.dt,
        )
        return s2, d

    _ = time_call("step + lidar (combo)", step_then_lidar, state, key, iters=15)

    # ---- Memory usage info ----
    print(f"\n=== Memory Usage ===")
    print(f"State size: {jax.tree_util.tree_flatten(state)[0][0].shape}")
    print(f"Lidar distances: {dists.shape}")

if __name__ == "__main__":
    main()