"""Simulation utilities for mobile robots and humans in indoor maps."""
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp

from .environment import IndoorMapBatch


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulator."""

    dt: float = 0.1
    robot_radius: float = 0.4
    person_radius: float = 0.35
    max_robot_speed: float = 1.0
    max_person_speed: float = 0.9
    person_noise_scale: float = 0.5


@dataclass
class AgentState:
    position: jnp.ndarray  # (..., 2)
    velocity: jnp.ndarray  # (..., 2)


@dataclass
class RobotState(AgentState):
    pass


@dataclass
class PeopleState(AgentState):
    pass


@dataclass
class SimulationState:
    robots: RobotState
    people: PeopleState


def _clip_speed(velocity: jnp.ndarray, max_speed: float) -> jnp.ndarray:
    speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
    safe_speed = jnp.where(speed > 1e-6, speed, 1.0)
    factor = jnp.minimum(1.0, max_speed / safe_speed)
    return velocity * factor


def _resolve_axis_aligned_collisions(
    positions: jnp.ndarray,
    radius: float,
    segments: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Push disc agents outside of walls represented by line segments."""

    def resolve_env(pos_env, seg_env, mask_env):
        radius_val = jnp.asarray(radius, dtype=pos_env.dtype)

        def collide_with_segment(pos, segment, is_valid):
            x0, y0, x1, y1 = segment
            is_vertical = jnp.isclose(x0, x1)
            is_horizontal = jnp.isclose(y0, y1)

            def vertical_case():
                xw = x0
                ymin = jnp.minimum(y0, y1)
                ymax = jnp.maximum(y0, y1)
                dx = pos[0] - xw
                within = (pos[1] >= ymin) & (pos[1] <= ymax)
                penetration = radius_val - jnp.abs(dx)
                penetration = jnp.maximum(0.0, penetration)
                push_dir = jnp.where(dx >= 0.0, 1.0, -1.0).astype(pos.dtype)
                delta = jnp.array([push_dir * penetration, 0.0], dtype=pos.dtype)
                return jnp.where(within, pos + delta, pos)

            def horizontal_case():
                yw = y0
                xmin = jnp.minimum(x0, x1)
                xmax = jnp.maximum(x0, x1)
                dy = pos[1] - yw
                within = (pos[0] >= xmin) & (pos[0] <= xmax)
                penetration = radius_val - jnp.abs(dy)
                penetration = jnp.maximum(0.0, penetration)
                push_dir = jnp.where(dy >= 0.0, 1.0, -1.0).astype(pos.dtype)
                delta = jnp.array([0.0, push_dir * penetration], dtype=pos.dtype)
                return jnp.where(within, pos + delta, pos)

            new_pos = jax.lax.cond(
                is_vertical,
                lambda _: vertical_case(),
                lambda _: jax.lax.cond(
                    is_horizontal, lambda __: horizontal_case(), lambda __: pos, None
                ),
                operand=None,
            )
            return jax.lax.select(is_valid, new_pos, pos)

        def resolve_single(pos):
            def scan_fn(p, data):
                segment, is_valid = data
                new_p = collide_with_segment(p, segment, is_valid)
                return new_p, None

            new_pos, _ = jax.lax.scan(scan_fn, pos, (seg_env, mask_env))
            return new_pos

        return jax.vmap(resolve_single)(pos_env)

    return jax.vmap(resolve_env)(positions, segments, mask)


def step_simulation(
    state: SimulationState,
    robot_actions: jnp.ndarray,
    map_batch: IndoorMapBatch,
    config: SimulationConfig,
    key: jax.Array,
) -> SimulationState:
    """Advance the simulation by one step."""

    dt = config.dt
    robot_vel = _clip_speed(robot_actions, config.max_robot_speed)
    robot_positions = state.robots.position + dt * robot_vel

    people_noise = config.person_noise_scale * jax.random.normal(key, state.people.velocity.shape)
    people_velocity = _clip_speed(state.people.velocity + people_noise, config.max_person_speed)
    people_positions = state.people.position + dt * people_velocity

    def resolve_positions(positions, radius):
        return _resolve_axis_aligned_collisions(positions, radius, map_batch.segments, map_batch.segment_mask)

    robot_positions = resolve_positions(robot_positions, config.robot_radius)
    people_positions = resolve_positions(people_positions, config.person_radius)

    robots = RobotState(position=robot_positions, velocity=robot_vel)
    people = PeopleState(position=people_positions, velocity=people_velocity)
    return SimulationState(robots=robots, people=people)


def lidar_scan(
    origin: jnp.ndarray,
    angles: jnp.ndarray,
    max_range: float,
    map_batch: IndoorMapBatch,
) -> jnp.ndarray:
    """Perform batched lidar ray casting."""

    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (num_rays, 2)

    def cast_environment(origin_env, segments_env, mask_env):
        vertical = jnp.isclose(segments_env[:, 0], segments_env[:, 2])
        horizontal = jnp.isclose(segments_env[:, 1], segments_env[:, 3])

        def cast_agent(origin_agent):
            def cast_ray(direction):
                ox, oy = origin_agent
                dx, dy = direction

                denom_v = jnp.where(jnp.abs(dx) < 1e-6, jnp.nan, dx)
                t_vertical = (segments_env[:, 0] - ox) / denom_v
                y_hit = oy + t_vertical * dy
                valid_v = (
                    mask_env
                    & vertical
                    & (t_vertical > 0.0)
                    & (y_hit >= jnp.minimum(segments_env[:, 1], segments_env[:, 3]))
                    & (y_hit <= jnp.maximum(segments_env[:, 1], segments_env[:, 3]))
                )

                denom_h = jnp.where(jnp.abs(dy) < 1e-6, jnp.nan, dy)
                t_horizontal = (segments_env[:, 1] - oy) / denom_h
                x_hit = ox + t_horizontal * dx
                valid_h = (
                    mask_env
                    & horizontal
                    & (t_horizontal > 0.0)
                    & (x_hit >= jnp.minimum(segments_env[:, 0], segments_env[:, 2]))
                    & (x_hit <= jnp.maximum(segments_env[:, 0], segments_env[:, 2]))
                )

                distances_v = jnp.where(valid_v, t_vertical, jnp.inf)
                distances_h = jnp.where(valid_h, t_horizontal, jnp.inf)
                min_distance = jnp.minimum(jnp.min(distances_v), jnp.min(distances_h))
                return jnp.minimum(min_distance, max_range)

            return jax.vmap(cast_ray)(directions)

        return jax.vmap(cast_agent)(origin_env)

    return jax.vmap(cast_environment)(origin, map_batch.segments, map_batch.segment_mask)

