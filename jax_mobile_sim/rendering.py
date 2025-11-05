"""Rendering utilities for optional simulator visualisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from .environment import IndoorMapBatch
from .simulator import SimulationState


@dataclass
class RenderConfig:
    """Configuration parameters for matplotlib rendering."""

    robot_radius: float = 0.2
    person_radius: float = 0.2
    lidar_color: str = "tab:orange"
    wall_color: str = "black"
    robot_color: str = "tab:blue"
    person_color: str = "tab:green"
    lidar_alpha: float = 0.4


def _to_numpy(array) -> np.ndarray:
    """Convert a JAX array or nested structure to a NumPy array."""

    return np.asarray(array)


def render_environment(
    state: SimulationState,
    maps: IndoorMapBatch,
    lidar_angles,
    lidar_distances,
    *,
    env_index: int = 0,
    robot_index: int = 0,
    config: Optional[RenderConfig] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Render a single environment using matplotlib."""

    if config is None:
        config = RenderConfig()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
        ax.clear()

    segments = _to_numpy(maps.segments[env_index])
    mask = _to_numpy(maps.segment_mask[env_index]).astype(bool)
    segments = segments[mask]
    world_size = _to_numpy(maps.world_size[env_index])

    if len(segments) > 0:
        walls = np.stack([segments[:, :2], segments[:, 2:]], axis=1)
        wall_collection = LineCollection(walls, colors=config.wall_color, linewidths=2.0)
        ax.add_collection(wall_collection)

    robot_positions = _to_numpy(state.robots.position[env_index])
    people_positions = _to_numpy(state.people.position[env_index])

    for person_pos in people_positions:
        circle = Circle(person_pos, radius=config.person_radius, color=config.person_color, alpha=0.8)
        ax.add_patch(circle)

    robot_pos = robot_positions[robot_index]
    robot_circle = Circle(robot_pos, radius=config.robot_radius, color=config.robot_color, alpha=0.9)
    ax.add_patch(robot_circle)

    directions = np.stack([np.cos(_to_numpy(lidar_angles)), np.sin(_to_numpy(lidar_angles))], axis=-1)
    distances = _to_numpy(lidar_distances[env_index, robot_index])
    endpoints = robot_pos + distances[:, None] * directions
    lidar_segments = np.stack([np.broadcast_to(robot_pos, endpoints.shape), endpoints], axis=1)
    lidar_collection = LineCollection(
        lidar_segments,
        colors=config.lidar_color,
        linewidths=1.0,
        alpha=config.lidar_alpha,
    )
    ax.add_collection(lidar_collection)

    ax.set_xlim(0.0, world_size[0])
    ax.set_ylim(0.0, world_size[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Environment {env_index} – Robot {robot_index}")
    ax.grid(False)
    fig.canvas.draw_idle()
    return ax

