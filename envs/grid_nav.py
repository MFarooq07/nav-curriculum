import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACTIONS = np.array([[ -1,  0],
                    [  1,  0],
                    [  0, -1],
                    [  0,  1]], dtype=np.int32)

RAY_DIRS = np.array([
    [-1,  0],  # N
    [-1,  1],  # NE
    [ 0,  1],  # E
    [ 1,  1],  # SE
    [ 1,  0],  # S
    [ 1, -1],  # SW
    [ 0, -1],  # W
    [-1,-1],   # NW
], dtype=np.int32)

@dataclass
class GridConfig:
    H: int = 15
    W: int = 15
    wall_prob: float = 0.18     # density of random walls
    max_steps: int = 200
    n_rays: int = 8
    ray_max: int = 10           # how far rays travel (cells)
    seed: int = 42

class GridNav:
    """
    Minimal 2D grid navigation with ray-distance observations.

    Observation (float32):
      [8 ray distances in [0,1], goal_dir_x, goal_dir_y]
    Reward:
      +1.0 on reaching goal
      -0.01 step cost each step
      -0.2 collision (moving into wall or out of bounds), episode terminates
    Done:
      True on goal, collision, or step_limit
    """
    def __init__(self, cfg: GridConfig = GridConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.H, self.W = cfg.H, cfg.W
        self.grid = None
        self.agent = None
        self.goal = None
        self.steps = 0

    # ---------- Public API ----------
    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._generate_map()
        self.steps = 0
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1
        a = ACTIONS[action]
        next_pos = self.agent + a
        reward = -0.01
        done = False
        info: Dict[str, Any] = {}

        # bounds or wall -> collision (terminate)
        if not self._in_bounds(next_pos) or self._is_wall(next_pos):
            reward += -1.0
            done = True
            info["event"] = "collision"
            obs = self._get_obs()
            return obs, float(reward), bool(done), info

        # move
        self.agent = next_pos

        # goal reached
        if np.array_equal(self.agent, self.goal):
            reward += 1.0
            done = True
            info["event"] = "goal"

        # step limit
        if self.steps >= self.cfg.max_steps:
            done = True
            info["event"] = info.get("event", "timeout")

        obs = self._get_obs()
        return obs, float(reward), bool(done), info

    # Optional pixel render hook for later days (returns HxW uint8 RGB)
    def render_rgb(self, scale: int = 8):
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        # walls
        img[self.grid == 1] = np.array([50, 50, 50], dtype=np.uint8)
        # free
        img[self.grid == 0] = np.array([220, 220, 220], dtype=np.uint8)
        # goal
        gx, gy = self.goal
        img[gx, gy] = np.array([0, 200, 0], dtype=np.uint8)
        # agent
        ax, ay = self.agent
        img[ax, ay] = np.array([200, 0, 0], dtype=np.uint8)
        if scale > 1:
            try:
                import cv2
                img = cv2.resize(img, (self.W*scale, self.H*scale), interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass
        return img

    # ---------- Internals ----------
    def _generate_map(self):
        # 0 = free, 1 = wall
        self.grid = (self.rng.random((self.H, self.W)) < self.cfg.wall_prob).astype(np.uint8)

        # keep outer border as walls to simplify rays
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # pick start/goal on free cells, ensure not same
        free = np.argwhere(self.grid == 0)
        if len(free) < 2:
            # fallback: clear center area
            self.grid[:, :] = 1
            self.grid[1:-1, 1:-1] = 0
            free = np.argwhere(self.grid == 0)

        idxs = self.rng.choice(len(free), size=2, replace=False)
        self.agent = free[idxs[0]]
        self.goal  = free[idxs[1]]

    def _get_obs(self):
        rays = self._ray_distances(self.agent)
        goal_vec = self.goal - self.agent
        norm = np.linalg.norm(goal_vec) + 1e-8
        goal_dir = (goal_vec / norm).astype(np.float32)
        obs = np.concatenate([rays, goal_dir], axis=0).astype(np.float32)
        return obs

    def _ray_distances(self, pos: np.ndarray):
        dists = []
        for d in RAY_DIRS[:self.cfg.n_rays]:
            step = 0
            p = pos.copy()
            hit = False
            while step < self.cfg.ray_max:
                p = p + d
                step += 1
                if not self._in_bounds(p) or self._is_wall(p):
                    hit = True
                    break
            # normalize to [0,1]; 1.0 means hit right away, 0.0 means max range free
            dist = step / self.cfg.ray_max
            dists.append(dist)
        return np.array(dists, dtype=np.float32)

    def _in_bounds(self, p: np.ndarray) -> bool:
        return 0 <= p[0] < self.H and 0 <= p[1] < self.W

    def _is_wall(self, p: np.ndarray) -> bool:
        return self.grid[p[0], p[1]] == 1

# Quick manual test
if __name__ == "__main__":
    env = GridNav(GridConfig())
    obs = env.reset(seed=123)
    print("obs shape:", obs.shape, "first obs:", obs[:5])
    done = False
    total = 0.0
    steps = 0
    while not done and steps < 10:
        a = np.random.randint(0, 4)
        obs, r, done, info = env.step(a)
        total += r
        steps += 1
    print("steps:", steps, "return:", total, "done:", done, "info:", info)