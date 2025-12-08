import numpy as np
from envs import GridNav, GridConfig
from algos.astar import astar

def test_astar_smoke():
    # Fixed small map with low wall density to likely have a path
    cfg = GridConfig(H=10, W=10, wall_prob=0.05, seed=777, max_steps=500)
    env = GridNav(cfg)
    env.reset()

    start = (int(env.agent[0]), int(env.agent[1]))
    goal  = (int(env.goal[0]),  int(env.goal[1]))

    path, expanded, rt = astar(env.grid, start, goal)
    assert path is not None, "A* failed to find a path when one likely exists"
    assert path[0] == start and path[-1] == goal
    # Path should not step into walls
    for x, y in path:
        assert env.grid[x, y] == 0
