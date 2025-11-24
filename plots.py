import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def moving_average(x, k=20):
    if k <= 1:
        return x
    return x.rolling(window=k, min_periods=1).mean()

def main():
    p = argparse.ArgumentParser(description="Plot success rate / returns from CSV")
    p.add_argument("--csv", type=str, required=True, help="path to metrics.csv")
    p.add_argument("--ma", type=int, default=20, help="moving average window")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = csv_path.parent

    df = pd.read_csv(csv_path)
    if "episode" not in df.columns or "success" not in df.columns or "return" not in df.columns:
        raise ValueError("metrics.csv must contain columns: episode, success, return")

    # Success rate (moving average of 0/1)
    plt.figure()
    plt.plot(df["episode"], moving_average(df["success"], args.ma))
    plt.title(f"Success Rate (MA={args.ma})")
    plt.xlabel("Episode")
    plt.ylabel("Success (moving average)")
    plt.tight_layout()
    succ_png = out_dir / "success_rate.png"
    plt.savefig(succ_png, dpi=150)
    plt.close()

    # Returns
    plt.figure()
    plt.plot(df["episode"], moving_average(df["return"], args.ma))
    plt.title(f"Return per Episode (MA={args.ma})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.tight_layout()
    ret_png = out_dir / "returns.png"
    plt.savefig(ret_png, dpi=150)
    plt.close()

    print(f"[OK] Wrote plots:\n - {succ_png}\n - {ret_png}")

if __name__ == "__main__":
    main()
