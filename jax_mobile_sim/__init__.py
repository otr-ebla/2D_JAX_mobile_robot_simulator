"""JAX-based 2D mobile robot simulator."""

from .environment import IndoorMap, IndoorMapBatch, MapGenerationConfig, generate_map_batch
from .simulator import (
    AgentState,
    PeopleState,
    RobotState,
    SimulationConfig,
    SimulationState,
    lidar_scan,
    step_simulation,
)

__all__ = [
    "AgentState",
    "PeopleState",
    "RobotState",
    "SimulationConfig",
    "SimulationState",
    "IndoorMap",
    "IndoorMapBatch",
    "MapGenerationConfig",
    "generate_map_batch",
    "lidar_scan",
    "step_simulation",
]
