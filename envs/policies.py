from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import heapq
import numpy as np

Coord = Tuple[int, int]  # (x, y) == (row, col)

class AStarFollower:
    """
    Grid A* planner for 4-connected motion on a 0/1 grid.
    plan(grid, start, goal) -> list of (x,y) cells to step to (excludes start).
    Returns [] if no path exists.
    """
    def __init__(self):
        # 4-dir moves: up, down, left, right
        self._dirs = [(-1,0), (1,0), (0,-1), (0,1)]

    def _in_bounds(self, grid: np.ndarray, x: int, y: int) -> bool:
        H, W = grid.shape[:2]
        return 0 <= x < H and 0 <= y < W

    def _is_free(self, grid: np.ndarray, x: int, y: int) -> bool:
        return grid[x, y] == 0

    def _neighbors(self, grid: np.ndarray, node: Coord) -> List[Coord]:
        x, y = node
        nbrs = []
        for dx, dy in self._dirs:
            nx, ny = x + dx, y + dy
            if self._in_bounds(grid, nx, ny) and self._is_free(grid, nx, ny):
                nbrs.append((nx, ny))
        return nbrs

    def _h(self, a: Coord, b: Coord) -> int:
        # Manhattan distance (admissible for 4-connected grids)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, grid: np.ndarray, start: Coord, goal: Coord) -> List[Coord]:
        if start == goal:
            return []

        # Priority queue entries: (f, g, tie, node)
        open_heap = []
        tie = 0
        g_score: Dict[Coord, int] = {start: 0}
        f_start = self._h(start, goal)
        heapq.heappush(open_heap, (f_start, 0, tie, start))

        came_from: Dict[Coord, Coord] = {}
        closed: set[Coord] = set()

        while open_heap:
            f, g, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                # reconstruct path (exclude start)
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                if path and path[0] == start:
                    path = path[1:]
                return path

            closed.add(current)

            for nb in self._neighbors(grid, current):
                if nb in closed:
                    continue
                tentative_g = g + 1
                if tentative_g < g_score.get(nb, 1_000_000_000):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    tie += 1
                    f_nb = tentative_g + self._h(nb, goal)
                    heapq.heappush(open_heap, (f_nb, tentative_g, tie, nb))

        # no path
        return []
