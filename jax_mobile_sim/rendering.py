"""Rendering utilities for optional simulator visualisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .environment import IndoorMapBatch
from .simulator import SimulationState


@dataclass
class RenderConfig:
    """Configuration parameters for matplotlib rendering."""

    robot_radius: float = 0.2
    person_radius: float = 0.2
    lidar_far_color: tuple[float, float, float] = (0.83, 0.83, 0.83)
    lidar_near_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    wall_color: str = "black"
    robot_color: str = "tab:blue"
    person_color: str = "tab:green"
    lidar_alpha: float = 0.4
    waypoint_marker_color: str = "red"
    waypoint_marker_size: float = 120.0
    heading_indicator_length: float = 0.15
    robot_heading_color: str = "white"
    person_heading_color: str = "white"
    heading_indicator_linewidth: float = 1.5


def _to_numpy(array) -> np.ndarray:
    """Convert a JAX array or nested structure to a NumPy array."""

    return np.asarray(array)


def _unit_heading(vector: np.ndarray) -> np.ndarray:
    heading = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(heading)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=float)
    return heading / norm


def render_environment(
    state: SimulationState,
    maps: IndoorMapBatch,
    lidar_angles,
    lidar_distances,
    *,
    env_index: int = 0,
    robot_index: int = 0,
    robot_waypoints=None,
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
    people_velocities = _to_numpy(state.people.velocity[env_index])

    for person_pos, person_vel in zip(people_positions, people_velocities):
        circle = Circle(person_pos, radius=config.person_radius, color=config.person_color, alpha=0.8)
        ax.add_patch(circle)
        heading = _unit_heading(person_vel)
        end = person_pos + heading * config.heading_indicator_length
        line = Line2D(
            [person_pos[0], end[0]],
            [person_pos[1], end[1]],
            color=config.person_heading_color,
            linewidth=config.heading_indicator_linewidth,
        )
        ax.add_line(line)

    robot_pos = robot_positions[robot_index]
    robot_velocity = _to_numpy(state.robots.velocity[env_index, robot_index])
    robot_circle = Circle(robot_pos, radius=config.robot_radius, color=config.robot_color, alpha=0.9)
    ax.add_patch(robot_circle)
    robot_heading = float(_to_numpy(state.robots.heading[env_index, robot_index]))
    heading_vec = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
    robot_end = robot_pos + heading_vec * config.heading_indicator_length
    heading = _unit_heading(robot_velocity)
    robot_end = robot_pos + heading * config.heading_indicator_length
    robot_line = Line2D(
        [robot_pos[0], robot_end[0]],
        [robot_pos[1], robot_end[1]],
        color=config.robot_heading_color,
        linewidth=config.heading_indicator_linewidth,
    )
    ax.add_line(robot_line)

    directions = np.stack([np.cos(_to_numpy(lidar_angles)), np.sin(_to_numpy(lidar_angles))], axis=-1)
    distances = _to_numpy(lidar_distances[env_index, robot_index])
    start_offsets = config.robot_radius * directions
    start_points = robot_pos + start_offsets
    visible_lengths = np.maximum(distances - config.robot_radius, 0.0)
    endpoints = start_points + visible_lengths[:, None] * directions
    lidar_segments = np.stack([start_points, endpoints], axis=1)
    if len(distances) > 0:
        max_distance = np.max(distances)
        if max_distance <= 1e-6:
            mix = np.zeros_like(distances)
        else:
            mix = 1.0 - np.clip(distances / max_distance, 0.0, 1.0)
        near_color = np.asarray((*config.lidar_near_color, config.lidar_alpha))
        far_color = np.asarray((*config.lidar_far_color, config.lidar_alpha))
        colors = far_color + mix[:, None] * (near_color - far_color)
    else:
        colors = np.zeros((0, 4))
    lidar_collection = LineCollection(
        lidar_segments,
        colors=colors,
        linewidths=1.0,
    )
    ax.add_collection(lidar_collection)

    if robot_waypoints is not None:
        waypoint = _to_numpy(robot_waypoints[env_index, robot_index])
        if np.all(np.isfinite(waypoint)):
            ax.scatter(
                waypoint[0],
                waypoint[1],
                marker="*",
                s=config.waypoint_marker_size,
                c=config.waypoint_marker_color,
                edgecolors="none",
            )

    ax.set_xlim(0.0, world_size[0])
    ax.set_ylim(0.0, world_size[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Environment {env_index} – Robot {robot_index}")
    ax.grid(False)
    fig.canvas.draw_idle()
    return ax

