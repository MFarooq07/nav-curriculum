import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def load_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def moving_average(x: np.ndarray, k: int = 20) -> np.ndarray:
    if len(x) == 0:
        return x
    k = max(1, k)
    c = np.cumsum(np.insert(x, 0, 0))
    ma = (c[k:] - c[:-k]) / float(k)
    # pad head with first valid value
    pad = np.full(k-1, ma[0] if len(ma) else 0.0)
    return np.concatenate([pad, ma]) if len(ma) else np.zeros_like(x)

def main():
    if len(sys.argv) < 2:
        print("Usage: python plots/plot_astar_metrics.py <results.csv>")
        sys.exit(1)

    path = sys.argv[1]
    rows = load_csv(path)
    if not rows:
        print("No data.")
        sys.exit(0)

    episodes = np.array([int(r["episode"]) for r in rows])
    success  = np.array([int(r["success"]) for r in rows], dtype=np.int32)

    # Success MA(20)
    ma = moving_average(success, k=20)

    plt.figure()
    plt.plot(episodes, ma, label="Success (MA=20)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (moving avg)")
    plt.title("A* Success Rate (MA=20)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print averages for A* metrics
    def to_float(x):
        try: return float(x)
        except: return np.nan

    path_lens = np.array([to_float(r["path_len"]) for r in rows], dtype=float)
    nodes_exp = np.array([to_float(r["nodes_expanded"]) for r in rows], dtype=float)

    # Filter NaNs if e.g., random policy rows used this plot
    path_lens = path_lens[~np.isnan(path_lens)]
    nodes_exp = nodes_exp[~np.isnan(nodes_exp)]

    if len(path_lens):
        print(f"Avg path length: {np.mean(path_lens):.2f}")
    if len(nodes_exp):
        print(f"Avg nodes expanded: {np.mean(nodes_exp):.2f}")

if __name__ == "__main__":
    main()
