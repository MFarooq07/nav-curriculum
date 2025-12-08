# --- in envs/policies.py ---
import heapq
import numpy as np

class AStarFollower:
    def __init__(self):
        pass

    def plan(self, grid: np.ndarray, start: tuple, goal: tuple):
        """
        grid: HxW uint8 array, 1=wall, 0=free
        start/goal: (x, y)
        returns: list of (x, y) cells from next step after start to goal (inclusive)
        """
        H, W = grid.shape
        def in_bounds(x, y): return 0 <= x < H and 0 <= y < W
        def passable(x, y):  return grid[x, y] == 0
        def neighbors(x, y):
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-neigh
                nx, ny = x+dx, y+dy
                if in_bounds(nx, ny) and passable(nx, ny):
                    yield (nx, ny)

        def h(a, b):  # Manhattan
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in neighbors(*current):
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + h(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return []  # no path

        # reconstruct path (exclude the start cell; include goal)
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path
