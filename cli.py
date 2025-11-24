import os, csv, json, time, argparse, numpy as np
from pathlib import Path

# Import your Day-2 env
from envs.grid_nav import GridNav, GridConfig, ACTIONS

def run_episode(env: GridNav, policy="random", seed=None):
    if seed is not None:
        obs = env.reset(seed=seed)
    else:
        obs = env.reset()
    done = False
    total = 0.0
    steps = 0
    while not done:
        if policy == "random":
            a = np.random.randint(0, len(ACTIONS))
        else:
            # place for future heuristic/learned policy
            a = np.random.randint(0, len(ACTIONS))
        obs, r, done, info = env.step(a)
        total += r
        steps += 1
    return steps, float(total), info

def main():
    p = argparse.ArgumentParser(description="GridNav rollouts → CSV logs")
    p.add_argument("--episodes", type=int, default=100, help="number of episodes")
    p.add_argument("--seed", type=int, default=123, help="base RNG seed")
    p.add_argument("--out", type=str, default="results", help="output root directory")
    p.add_argument("--wall_prob", type=float, default=None, help="override cfg.wall_prob")
    p.add_argument("--H", type=int, default=None, help="override cfg.H")
    p.add_argument("--W", type=int, default=None, help="override cfg.W")
    p.add_argument("--ray_max", type=int, default=None, help="override cfg.ray_max")
    args = p.parse_args()

    # Build config (overrides optional)
    cfg = GridConfig()
    if args.wall_prob is not None: cfg.wall_prob = args.wall_prob
    if args.H is not None: cfg.H = args.H
    if args.W is not None: cfg.W = args.W
    if args.ray_max is not None: cfg.ray_max = args.ray_max

    # Make output run dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config as JSON for reproducibility
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "episodes": args.episodes,
            "seed": args.seed,
            "grid_config": {
                "H": cfg.H, "W": cfg.W, "wall_prob": cfg.wall_prob,
                "max_steps": cfg.max_steps, "n_rays": cfg.n_rays, "ray_max": cfg.ray_max, "seed": cfg.seed
            }
        }, f, indent=2)

    # Prepare env and RNG
    np.random.seed(args.seed)
    env = GridNav(cfg)

    # CSV header
    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode","steps","return","event","success","episode_seed"])

        for ep in range(1, args.episodes + 1):
            ep_seed = int(np.random.randint(0, 2**31-1))
            steps, ret, info = run_episode(env, seed=ep_seed)
            event = info.get("event", "")
            success = 1 if event == "goal" else 0
            w.writerow([ep, steps, f"{ret:.6f}", event, success, ep_seed])

    print(f"[OK] Wrote: {csv_path}")
    print("[TIP] Next: python plots.py --csv", csv_path)

if __name__ == "__main__":
    main()
