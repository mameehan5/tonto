from gymnasium import spaces

import numpy as np

def obs_to_key(obs, space):
    """
    Convert a Gym observation into a hashable key,
    following the structure of its observation space.
    """
    if isinstance(space, spaces.Dict):
        return tuple(
            obs_to_key(obs[k], space.spaces[k])
            for k in sorted(space.spaces.keys())
        )

    elif isinstance(space, spaces.Discrete):
        return int(obs)

    elif isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete)):
        return tuple(np.asarray(obs, dtype=int).flatten())

    elif isinstance(space, spaces.Box):
        return tuple(np.asarray(obs).flatten())

    else:
        raise TypeError(f"Unsupported space type: {type(space)}")