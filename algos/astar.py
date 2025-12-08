import time
import heapq
import numpy as np
from typing import List, Optional, Tuple, Dict

# 4-connected moves (match your env ACTIONS mapping)
# 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
MOVE_DIRS = np.array([
    [-1,  0],  # up
    [ 1,  0],  # down
    [ 0, -1],  # left
    [ 0,  1],  # right
], dtype=np.int32)

def _in_bounds(p: Tuple[int, int], H: int, W: int) -> bool:
    return 0 <= p[0] < H and 0 <= p[1] < W

def _is_wall(grid: np.ndarray, p: Tuple[int, int]) -> bool:
    return grid[p[0], p[1]] == 1

def _neighbors(grid: np.ndarray, p: Tuple[int, int]) -> List[Tuple[int, int]]:
    H, W = grid.shape
    nbrs = []
    for d in MOVE_DIRS:
        nx, ny = p[0] + int(d[0]), p[1] + int(d[1])
        if _in_bounds((nx, ny), H, W) and not _is_wall(grid, (nx, ny)):
            nbrs.append((nx, ny))
    return nbrs

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Tuple[Optional[List[Tuple[int, int]]], int, float]:
    """
    Run A* on a 4-connected grid.
    Returns: (path or None, nodes_expanded, runtime_ms)
      path includes start..goal (both endpoints)
    """
    t0 = time.perf_counter()

    if _is_wall(grid, start) or _is_wall(grid, goal):
        return None, 0, (time.perf_counter() - t0) * 1000.0

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_cost: Dict[Tuple[int, int], float] = {start: 0.0}

    nodes_expanded = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        nodes_expanded += 1

        if current == goal:
            # reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return path, nodes_expanded, dt_ms

        for nb in _neighbors(grid, current):
            tentative_g = g_cost[current] + 1.0  # grid step cost
            if tentative_g < g_cost.get(nb, float("inf")):
                came_from[nb] = current
                g_cost[nb] = tentative_g
                # A* priority = g + h (Manhattan), tie-break small epsilon on g
                f = tentative_g + _manhattan(nb, goal) + 1e-6 * tentative_g
                heapq.heappush(open_heap, (f, nb))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return None, nodes_expanded, dt_ms
