from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from .grid_nav import GridNav, GridConfig

class _Discrete:
    def __init__(self, n:int): self.n=n
    def sample(self, rng=None):
        if rng is None: rng=np.random
        return int(rng.randint(0,self.n))
    def __repr__(self): return f"Discrete({self.n})"

class _Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, shape, dtype
    def sample(self, rng=None):
        if rng is None: rng=np.random
        return rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)
    def __repr__(self): return f"Box(low={self.low}, high={self.high}, shape={self.shape}, dtype={self.dtype})"

@dataclass
class GridNavSpec:
    cfg: GridConfig
    action_space: _Discrete
    observation_space: _Box

def make_gridnav(cfg: GridConfig = GridConfig()) -> Tuple[GridNav, GridNavSpec]:
    env = GridNav(cfg)
    obs_dim = cfg.n_rays + 2  # 8 rays + 2D goal dir
    spec = GridNavSpec(
        cfg=cfg,
        action_space=_Discrete(4),
        observation_space=_Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
    )
    return env, spec
