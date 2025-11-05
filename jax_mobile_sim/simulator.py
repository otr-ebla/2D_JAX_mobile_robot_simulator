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
    robot_radius: float = 0.2
    person_radius: float = 0.2
    max_robot_speed: float = 1.0
    max_robot_angular_speed: float = 1.5
    max_person_speed: float = 0.9
    person_noise_scale: float = 0.5


@dataclass
class AgentState:
    position: jnp.ndarray  # (..., 2)
    velocity: jnp.ndarray  # (..., 2)


@dataclass
class RobotState(AgentState):
    heading: jnp.ndarray  # (...,)


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


def _wrap_angle(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to [-pi, pi)."""

    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


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
    robot_linear_cmd = jnp.clip(
        robot_actions[..., 0], -config.max_robot_speed, config.max_robot_speed
    )
    robot_angular_cmd = jnp.clip(
        robot_actions[..., 1], -config.max_robot_angular_speed, config.max_robot_angular_speed
    )

    headings = state.robots.heading
    heading_dirs = jnp.stack([jnp.cos(headings), jnp.sin(headings)], axis=-1)
    robot_vel = robot_linear_cmd[..., None] * heading_dirs
    robot_positions = state.robots.position + dt * robot_vel
    new_headings = _wrap_angle(headings + dt * robot_angular_cmd)

    people_noise = config.person_noise_scale * jax.random.normal(key, state.people.velocity.shape)
    people_velocity = _clip_speed(state.people.velocity + people_noise, config.max_person_speed)
    people_positions = state.people.position + dt * people_velocity

    def resolve_positions(positions, radius):
        return _resolve_axis_aligned_collisions(positions, radius, map_batch.segments, map_batch.segment_mask)

    robot_positions = resolve_positions(robot_positions, config.robot_radius)
    people_positions = resolve_positions(people_positions, config.person_radius)

    heading_dirs_out = jnp.stack([jnp.cos(new_headings), jnp.sin(new_headings)], axis=-1)
    robot_velocity_out = robot_linear_cmd[..., None] * heading_dirs_out
    robots = RobotState(position=robot_positions, velocity=robot_velocity_out, heading=new_headings)
    people = PeopleState(position=people_positions, velocity=people_velocity)
    return SimulationState(robots=robots, people=people)


def lidar_scan(
    origin: jnp.ndarray,
    angles: jnp.ndarray,
    max_range: float,
    map_batch: IndoorMapBatch,
    *,
    people_positions: jnp.ndarray | None = None,
    person_radius: float = 0.0,
) -> jnp.ndarray:
    """Perform batched lidar ray casting."""

    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (num_rays, 2)
    max_range = jnp.minimum(jnp.asarray(max_range, dtype=origin.dtype), 30.0)

    if people_positions is None:
        people_positions = jnp.zeros((origin.shape[0], 0, 2), dtype=origin.dtype)

    person_radius_sq = jnp.asarray(person_radius, dtype=origin.dtype) ** 2

    def cast_environment(origin_env, segments_env, mask_env, people_env):
        mask_env = mask_env.astype(bool)
        x0, y0, x1, y1 = jnp.moveaxis(segments_env, -1, 0)
        vertical = jnp.isclose(x0, x1)
        horizontal = jnp.isclose(y0, y1)
        x_min = jnp.minimum(x0, x1)
        x_max = jnp.maximum(x0, x1)
        y_min = jnp.minimum(y0, y1)
        y_max = jnp.maximum(y0, y1)

        num_people = people_env.shape[0]
        ray_dx = directions[:, 0]
        ray_dy = directions[:, 1]
        ray_norm_sq = jnp.sum(directions * directions, axis=-1)

        def cast_agent(origin_agent):
            ox, oy = origin_agent

            valid_dx = jnp.abs(ray_dx) > 1e-6
            safe_dx = jnp.where(valid_dx, ray_dx, 1.0)
            t_vertical = (x0[None, :] - ox) / safe_dx[:, None]
            y_hit = oy + t_vertical * ray_dy[:, None]
            vertical_hits = (
                mask_env[None, :]
                & vertical[None, :]
                & valid_dx[:, None]
                & (t_vertical > 0.0)
                & (y_hit >= y_min[None, :])
                & (y_hit <= y_max[None, :])
            )
            distances_v = jnp.where(vertical_hits, t_vertical, jnp.inf)

            valid_dy = jnp.abs(ray_dy) > 1e-6
            safe_dy = jnp.where(valid_dy, ray_dy, 1.0)
            t_horizontal = (y0[None, :] - oy) / safe_dy[:, None]
            x_hit = ox + t_horizontal * ray_dx[:, None]
            horizontal_hits = (
                mask_env[None, :]
                & horizontal[None, :]
                & valid_dy[:, None]
                & (t_horizontal > 0.0)
                & (x_hit >= x_min[None, :])
                & (x_hit <= x_max[None, :])
            )
            distances_h = jnp.where(horizontal_hits, t_horizontal, jnp.inf)

            min_walls = jnp.minimum(
                jnp.min(distances_v, axis=1), jnp.min(distances_h, axis=1)
            )

            def compute_people_distances():
                oc = origin_agent - people_env  # (num_people, 2)
                c = jnp.sum(oc * oc, axis=-1) - person_radius_sq
                b = 2.0 * jnp.einsum("rd,pd->rp", directions, oc)
                discriminant = b * b - 4.0 * ray_norm_sq[:, None] * c[None, :]
                sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
                inv_two_a = 0.5 / jnp.maximum(ray_norm_sq, 1e-12)
                t0 = (-b - sqrt_disc) * inv_two_a[:, None]
                t1 = (-b + sqrt_disc) * inv_two_a[:, None]
                candidates = jnp.stack([t0, t1], axis=-1)
                valid = (discriminant >= 0.0)[..., None] & (candidates >= 0.0)
                distances = jnp.where(valid, candidates, jnp.inf)
                min_people = jnp.min(jnp.min(distances, axis=-1), axis=-1)
                return jnp.minimum(min_walls, min_people)

            min_distance = jax.lax.cond(
                num_people > 0,
                lambda _: compute_people_distances(),
                lambda _: min_walls,
                operand=None,
            )

            return jnp.minimum(min_distance, max_range)

        return jax.vmap(cast_agent)(origin_env)

    return jax.vmap(cast_environment)(
        origin, map_batch.segments, map_batch.segment_mask, people_positions
    )

