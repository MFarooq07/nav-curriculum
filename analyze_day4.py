import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # expected columns: episode, steps, return, event
            r["episode"] = int(r["episode"])
            r["steps"] = int(r["steps"])
            r["return"] = float(r["return"])
            rows.append(r)
    rows.sort(key=lambda x: x["episode"])
    return rows

def moving_average(x, k):
    x = np.asarray(x, dtype=float)
    if k <= 1:
        return x
    # keep same length as x
    return np.convolve(x, np.ones(k)/k, mode="same")

def summarize(rows, name):
    events = [r["event"] for r in rows]
    steps  = np.array([r["steps"] for r in rows])
    success = np.array([e == "goal" for e in events])
    collisions = np.sum([e == "collision" for e in events])
    timeouts   = np.sum([e == "timeout" for e in events])

    succ_rate = success.mean()
    succ_steps = steps[success] if success.any() else np.array([])
    succ_steps_mean = succ_steps.mean() if succ_steps.size else float("nan")
    succ_steps_median = np.median(succ_steps) if succ_steps.size else float("nan")

    print(f"\n== {name} ==")
    print(f"Episodes: {len(rows)}")
    print(f"Success rate: {succ_rate*100:.1f}%")
    print(f"Collisions: {collisions} | Timeouts: {timeouts}")
    print(f"Steps on successes — mean: {succ_steps_mean:.2f}, median: {succ_steps_median:.2f}")

    return {
        "succ_rate": succ_rate,
        "collisions": collisions,
        "timeouts": timeouts,
        "succ_steps": succ_steps
    }

def plot_success_series(rows, out_png, ma=20, title="Success Rate (per-episode)"):
    y = np.array([1.0 if r["event"] == "goal" else 0.0 for r in rows], dtype=float)
    ma_y = moving_average(y, ma)
    xs = np.arange(1, len(rows)+1)

    plt.figure(figsize=(9,4.5))
    plt.plot(xs, y, alpha=0.2, label="Per-episode (0/1)")
    plt.plot(xs, ma_y, linewidth=2, label=f"Moving Avg (k={ma})")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title(title + f" — MA={ma}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_steps_hist(rows, out_png, title="Steps on Successful Episodes"):
    succ_steps = [r["steps"] for r in rows if r["event"] == "goal"]
    if not succ_steps:
        print("No successes — skipping steps histogram.")
        return
    plt.figure(figsize=(6,4))
    plt.hist(succ_steps, bins=30)
    plt.xlabel("Steps")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random_csv", type=Path, required=True)
    ap.add_argument("--astar_csv",  type=Path, required=True)
    ap.add_argument("--out_dir",    type=Path, default=Path("results"))
    ap.add_argument("--ma", type=int, default=20, help="Moving average window")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rnd = load_csv(args.random_csv)
    ast = load_csv(args.astar_csv)

    rnd_stats = summarize(rnd, "Random")
    ast_stats = summarize(ast, "A*")

    # Plots
    plot_success_series(rnd, args.out_dir/"day4_random_success_ma.png", ma=args.ma,
                        title="Random: Success vs Episode")
    plot_success_series(ast, args.out_dir/"day4_astar_success_ma.png", ma=args.ma,
                        title="A*: Success vs Episode")

    plot_steps_hist(rnd, args.out_dir/"day4_random_steps_hist.png",
                    title="Random: Steps on Successful Episodes")
    plot_steps_hist(ast, args.out_dir/"day4_astar_steps_hist.png",
                    title="A*: Steps on Successful Episodes")

if __name__ == "__main__":
    main()
