from dataclasses import is_dataclass, fields
from jax import tree_util

def register_dataclass_as_pytree(cls):
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    names = [f.name for f in fields(cls)]
    def flatten(obj):
        children = tuple(getattr(obj, n) for n in names)
        return children, None
    def unflatten(aux, children):
        return cls(**{n: v for n, v in zip(names, children)})
    tree_util.register_pytree_node(cls, flatten, unflatten)

try:
    # register your dataclasses
    from jax_mobile_sim.simulator import SimulationState, RobotState, PeopleState
    from jax_mobile_sim.environment import IndoorMapBatch
    for cls in (RobotState, PeopleState, SimulationState, IndoorMapBatch):
        register_dataclass_as_pytree(cls)
except Exception:
    # avoid import errors during builds
    pass
