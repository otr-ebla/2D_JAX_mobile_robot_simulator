"""Environment generation utilities for 2D indoor maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass
class MapGenerationConfig:
    """Configuration for procedurally generated indoor maps."""

    world_size: Tuple[float, float] = (20.0, 12.0)
    min_rooms: Tuple[int, int] = (2, 2)
    max_rooms: Tuple[int, int] = (4, 3)
    corridor_probability: float = 0.25
    min_door_width: float = 1.0
    max_door_width: float = 2.5
    max_segments: int = 160


@dataclass
class IndoorMap:
    """Single indoor map with padded wall segments."""

    segments: jnp.ndarray  # (max_segments, 4) => (x1, y1, x2, y2)
    segment_mask: jnp.ndarray  # (max_segments,)
    world_size: jnp.ndarray  # (2,)


@dataclass
class IndoorMapBatch:
    """Batch of indoor maps for vectorized simulation."""

    segments: jnp.ndarray  # (batch, max_segments, 4)
    segment_mask: jnp.ndarray  # (batch, max_segments)
    world_size: jnp.ndarray  # (batch, 2)


def _add_axis_aligned_wall(
    segments: list,
    start: Tuple[float, float],
    end: Tuple[float, float],
    door_keys: jax.Array,
    config: MapGenerationConfig,
    *,
    allow_doors: bool = True,
):
    """Append axis aligned wall segments with optional random doors."""

    if start[0] == end[0]:  # vertical wall
        y0, y1 = (start[1], end[1]) if start[1] <= end[1] else (end[1], start[1])
        wall_length = y1 - y0
        door_positions = []
        if allow_doors and wall_length > config.min_door_width * 1.5:
            door_count = int(jax.random.randint(door_keys[0], (), 0, 3))
            base_width = float(
                jax.random.uniform(
                    door_keys[1], (), minval=config.min_door_width, maxval=config.max_door_width
                )
            )
            door_width = float(min(base_width, 0.5 * wall_length))
            for i in range(door_count):
                key = jax.random.fold_in(door_keys[2], i)
                lo = y0 + door_width
                hi = y1 - door_width
                if hi <= lo:
                    break
                door_center = float(jax.random.uniform(key, (), minval=lo, maxval=hi))
                door_positions.append((door_center - 0.5 * door_width, door_center + 0.5 * door_width))
        segments.extend(_segment_with_openings((start[0], y0), (end[0], y1), door_positions, vertical=True))
    elif start[1] == end[1]:  # horizontal wall
        x0, x1 = (start[0], end[0]) if start[0] <= end[0] else (end[0], start[0])
        wall_length = x1 - x0
        door_positions = []
        if allow_doors and wall_length > config.min_door_width * 1.5:
            door_count = int(jax.random.randint(door_keys[0], (), 0, 3))
            base_width = float(
                jax.random.uniform(
                    door_keys[1], (), minval=config.min_door_width, maxval=config.max_door_width
                )
            )
            door_width = float(min(base_width, 0.5 * wall_length))
            for i in range(door_count):
                key = jax.random.fold_in(door_keys[2], i)
                lo = x0 + door_width
                hi = x1 - door_width
                if hi <= lo:
                    break
                door_center = float(jax.random.uniform(key, (), minval=lo, maxval=hi))
                door_positions.append((door_center - 0.5 * door_width, door_center + 0.5 * door_width))
        segments.extend(
            _segment_with_openings((x0, start[1]), (x1, end[1]), door_positions, vertical=False)
        )
    else:
        raise ValueError("Only axis-aligned walls are supported")


def _segment_with_openings(
    start: Tuple[float, float],
    end: Tuple[float, float],
    openings: list[Tuple[float, float]],
    *,
    vertical: bool,
):
    """Split a wall into multiple segments according to the openings."""

    openings = sorted(openings, key=lambda rng: rng[0])
    output = []
    cursor = start[1] if vertical else start[0]
    extent = end[1] if vertical else end[0]
    for low, high in openings:
        if low > cursor:
            if vertical:
                output.append((start[0], cursor, end[0], low))
            else:
                output.append((cursor, start[1], low, end[1]))
        cursor = max(cursor, high)
    if cursor < extent:
        if vertical:
            output.append((start[0], cursor, end[0], extent))
        else:
            output.append((cursor, start[1], extent, end[1]))
    return [seg for seg in output if max(abs(seg[0] - seg[2]), abs(seg[1] - seg[3])) > 1e-5]


def generate_map(key: jax.Array, config: MapGenerationConfig) -> IndoorMap:
    """Generate a single indoor map."""

    segments = []
    world_w, world_h = config.world_size

    # Outer boundary
    _add_axis_aligned_wall(segments, (0.0, 0.0), (world_w, 0.0), jax.random.split(key, 3), config, allow_doors=False)
    _add_axis_aligned_wall(
        segments,
        (0.0, world_h),
        (world_w, world_h),
        jax.random.split(jax.random.fold_in(key, 1), 3),
        config,
        allow_doors=False,
    )
    _add_axis_aligned_wall(
        segments,
        (0.0, 0.0),
        (0.0, world_h),
        jax.random.split(jax.random.fold_in(key, 2), 3),
        config,
        allow_doors=False,
    )
    _add_axis_aligned_wall(
        segments,
        (world_w, 0.0),
        (world_w, world_h),
        jax.random.split(jax.random.fold_in(key, 3), 3),
        config,
        allow_doors=False,
    )

    # Determine grid layout
    key_x, key_y, key_split = jax.random.split(jax.random.fold_in(key, 4), 3)
    nx = int(
        jax.random.randint(key_x, (), minval=config.min_rooms[0], maxval=config.max_rooms[0] + 1)
    )
    ny = int(
        jax.random.randint(key_y, (), minval=config.min_rooms[1], maxval=config.max_rooms[1] + 1)
    )

    def random_partitions(total: float, parts: int, key: jax.Array) -> list[float]:
        if parts <= 0:
            return []
        alpha = jnp.ones(parts)
        weights = jax.random.dirichlet(key, alpha)
        return [float(total * w) for w in weights]

    widths = random_partitions(world_w, nx, key_split)
    heights = random_partitions(world_h, ny, jax.random.fold_in(key_split, 1))

    x_positions = [0.0]
    for w in widths[:-1]:
        x_positions.append(x_positions[-1] + w)
    y_positions = [0.0]
    for h in heights[:-1]:
        y_positions.append(y_positions[-1] + h)

    # Interior walls
    corridor_thresh = config.corridor_probability
    for i, x in enumerate(x_positions[1:], start=1):
        # Optionally skip wall to create corridor
        is_corridor = bool(jax.random.bernoulli(jax.random.fold_in(key, 10 + i), corridor_thresh))
        if is_corridor:
            continue
        _add_axis_aligned_wall(
            segments,
            (x, 0.0),
            (x, world_h),
            jax.random.split(jax.random.fold_in(key, 20 + i), 3),
            config,
        )
    for j, y in enumerate(y_positions[1:], start=1):
        is_corridor = bool(jax.random.bernoulli(jax.random.fold_in(key, 40 + j), corridor_thresh))
        if is_corridor:
            continue
        _add_axis_aligned_wall(
            segments,
            (0.0, y),
            (world_w, y),
            jax.random.split(jax.random.fold_in(key, 50 + j), 3),
            config,
        )

    # Pad to fixed size
    max_segments = config.max_segments
    if len(segments) > max_segments:
        segments = segments[:max_segments]
    padded = segments + [(0.0, 0.0, 0.0, 0.0)] * (max_segments - len(segments))
    mask = [True] * min(len(segments), max_segments) + [False] * (max_segments - min(len(segments), max_segments))

    return IndoorMap(
        segments=jnp.array(padded, dtype=jnp.float32),
        segment_mask=jnp.array(mask, dtype=bool),
        world_size=jnp.array(config.world_size, dtype=jnp.float32),
    )


def generate_map_batch(key: jax.Array, batch_size: int, config: MapGenerationConfig) -> IndoorMapBatch:
    """Generate a batch of maps with independent randomness."""

    keys = jax.random.split(key, batch_size)
    maps = [generate_map(k, config) for k in keys]
    segments = jnp.stack([m.segments for m in maps], axis=0)
    mask = jnp.stack([m.segment_mask for m in maps], axis=0)
    sizes = jnp.stack([m.world_size for m in maps], axis=0)
    return IndoorMapBatch(segments=segments, segment_mask=mask, world_size=sizes)

