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
    dynamics_substeps: int = 4
    lidar_updates_per_step: int = 12


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
    """Advance the simulation by one step using substepped dynamics."""

    num_substeps = max(int(config.dynamics_substeps), 1)
    dt_total = config.dt
    dt = dt_total / num_substeps
    robot_linear_cmd = jnp.clip(
        robot_actions[..., 0], -config.max_robot_speed, config.max_robot_speed
    )
    robot_angular_cmd = jnp.clip(
        robot_actions[..., 1], -config.max_robot_angular_speed, config.max_robot_angular_speed
    )

    def resolve_positions(positions, radius):
        return _resolve_axis_aligned_collisions(
            positions, radius, map_batch.segments, map_batch.segment_mask
        )

    noise_scale = config.person_noise_scale * jnp.sqrt(
        dt / jnp.maximum(dt_total, 1e-6)
    )

    def fori_body(_, carry):
        robot_positions, headings, people_positions, people_velocity, key_inner = carry
        key_inner, noise_key = jax.random.split(key_inner)

        heading_dirs = jnp.stack([jnp.cos(headings), jnp.sin(headings)], axis=-1)
        robot_positions = robot_positions + dt * (robot_linear_cmd[..., None] * heading_dirs)
        headings = _wrap_angle(headings + dt * robot_angular_cmd)

        people_noise = noise_scale * jax.random.normal(noise_key, people_velocity.shape)
        people_velocity = _clip_speed(people_velocity + people_noise, config.max_person_speed)
        people_positions = people_positions + dt * people_velocity

        robot_positions = resolve_positions(robot_positions, config.robot_radius)
        people_positions = resolve_positions(people_positions, config.person_radius)

        return robot_positions, headings, people_positions, people_velocity, key_inner

    robot_positions, headings, people_positions, people_velocity, final_key = jax.lax.fori_loop(
        0,
        num_substeps,
        fori_body,
        (
            state.robots.position,
            state.robots.heading,
            state.people.position,
            state.people.velocity,
            key,
        ),
    )
    del final_key

    heading_dirs_out = jnp.stack([jnp.cos(headings), jnp.sin(headings)], axis=-1)
    robot_velocity_out = robot_linear_cmd[..., None] * heading_dirs_out
    robots = RobotState(position=robot_positions, velocity=robot_velocity_out, heading=headings)
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
    num_subsamples: int = 1,
    origin_velocities: jnp.ndarray | None = None,
    people_velocities: jnp.ndarray | None = None,
    dt: float | None = None,
    return_history: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Perform batched lidar ray casting with optional high-frequency updates."""

    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (num_rays, 2)
    max_range_val = jnp.minimum(jnp.asarray(max_range, dtype=origin.dtype), 30.0)

    if people_positions is None:
        people_positions = jnp.zeros((origin.shape[0], 0, 2), dtype=origin.dtype)
    if origin_velocities is None:
        origin_velocities = jnp.zeros_like(origin)
    if people_velocities is None:
        people_velocities = jnp.zeros_like(people_positions)

    origin_velocities = jnp.asarray(origin_velocities, dtype=origin.dtype)
    people_velocities = jnp.asarray(people_velocities, dtype=origin.dtype)

    person_radius_sq = jnp.asarray(person_radius, dtype=origin.dtype) ** 2
    segments = map_batch.segments
    segment_mask = map_batch.segment_mask

    ray_dx = directions[:, 0]
    ray_dy = directions[:, 1]
    ray_norm_sq = jnp.sum(directions * directions, axis=-1)

    def cast_environment(origin_env, segments_env, mask_env, people_env):
        mask_env = mask_env.astype(bool)
        x0, y0, x1, y1 = jnp.moveaxis(segments_env, -1, 0)
        vertical = jnp.isclose(x0, x1)
        horizontal = jnp.isclose(y0, y1)
        x_min = jnp.minimum(x0, x1)
        x_max = jnp.maximum(x0, x1)
        y_min = jnp.minimum(y0, y1)
        y_max = jnp.maximum(y0, y1)
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

            if people_env.shape[0] > 0:
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
                min_distance = jnp.minimum(min_walls, min_people)
            else:
                min_distance = min_walls

            return jnp.minimum(min_distance, max_range_val)

        return jax.vmap(cast_agent)(origin_env)

    def cast_snapshot(origin_snapshot, people_snapshot):
        return jax.vmap(cast_environment)(
            origin_snapshot, segments, segment_mask, people_snapshot
        )

    samples = max(int(num_subsamples), 1)
    if samples == 1:
        distances = cast_snapshot(origin, people_positions)
        if return_history:
            history = distances[jnp.newaxis, ...]
            return distances, history
        return distances

    total_dt = 0.0 if dt is None else float(dt)
    sub_dt = jnp.asarray(total_dt / samples, dtype=origin.dtype)
    origin_delta = origin_velocities * sub_dt
    people_delta = people_velocities * sub_dt

    def scan_step(carry, _):
        origin_pos, people_pos = carry
        distances = cast_snapshot(origin_pos, people_pos)
        next_origin = origin_pos + origin_delta
        next_people = people_pos + people_delta
        return (next_origin, next_people), distances

    (_, _), history = jax.lax.scan(
        scan_step, (origin, people_positions), None, length=samples
    )
    final = history[-1]
    if return_history:
        return final, history
    return final

