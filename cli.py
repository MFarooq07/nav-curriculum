# cli.py
import argparse
import csv
from pathlib import Path
import numpy as np

from envs.grid_nav import GridNav, GridConfig   # ensure file is envs/gridnav.py
from envs.policies import AStarFollower        # ensure file is envs/policies.py

ACTIONS = np.array([[ -1,  0],
                    [  1,  0],
                    [  0, -1],
                    [  0,  1]], dtype=np.int32)

def run_episode(env: GridNav, policy: str):
    obs = env.reset()
    done = False
    steps = 0
    total = 0.0
    event = "timeout"

    # Prepare A* path (list of (x,y)) if requested
    path = []
    if policy == "astar":
        follower = AStarFollower()
        start = tuple(env.agent.tolist())
        goal  = tuple(env.goal.tolist())
        grid  = env.grid.copy()
        path = follower.plan(grid, start, goal)  # may return []

    while not done:
        if policy == "random":
            a = int(np.random.randint(0, 4))
        elif policy == "astar":
            if path:
                nx, ny = path.pop(0)
                ax, ay = env.agent
                dx, dy = (nx - ax, ny - ay)
                if   (dx, dy) == (-1, 0): a = 0
                elif (dx, dy) == ( 1, 0): a = 1
                elif (dx, dy) == ( 0,-1): a = 2
                elif (dx, dy) == ( 0, 1): a = 3
                else:
                    # Path discontinuity: stop moving (or choose random)
                    a = 0
            else:
                # No path available or finished following it
                a = 0
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, r, done, info = env.step(a)
        total += r
        steps += 1
        if done:
            event = info.get("event", "timeout")

    return {"steps": steps, "return": float(total), "event": event}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "astar"], required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, required=True)

    # env config
    ap.add_argument("--wall_prob", type=float, default=0.18)
    ap.add_argument("--H", type=int, default=15)
    ap.add_argument("--W", type=int, default=15)
    ap.add_argument("--ray_max", type=int, default=10)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = GridConfig(H=args.H, W=args.W, wall_prob=args.wall_prob, ray_max=args.ray_max, seed=args.seed)
    env = GridNav(cfg)

    # Write CSV
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "policy", "steps", "return", "event"])
        writer.writeheader()

        for ep in range(args.episodes):
            row = run_episode(env, args.policy)
            writer.writerow({
                "episode": ep,
                "policy": args.policy,
                "steps": row["steps"],
                "return": row["return"],
                "event": row["event"],
            })

    print(f"✅ Wrote results to: {out_path}")

if __name__ == "__main__":
    main()
