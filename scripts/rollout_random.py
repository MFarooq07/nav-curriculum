import argparse, os, time
import numpy as np

from envs.grid_nav import GridConfig
from envs.wrappers import make_gridnav

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--record", type=int, default=0, help="1 to save GIFs (requires imageio)")
    p.add_argument("--outdir", type=str, default="results/day3")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    env, spec = make_gridnav(GridConfig(seed=args.seed))
    rng = np.random.default_rng(args.seed)

    returns = []
    lengths = []
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        frames = []
        done = False
        ret, steps = 0.0, 0
        while not done:
            a = spec.action_space.sample(rng)
            obs, r, done, info = env.step(a)
            ret += r
            steps += 1
            if args.record:
                frames.append(env.render_rgb(scale=8))
        returns.append(ret)
        lengths.append(steps)

        print(f"ep {ep+1}/{args.episodes}: steps={steps} return={ret:.3f} info={info}")

        if args.record and len(frames)>0:
            try:
                import imageio.v2 as imageio
                fn = os.path.join(args.outdir, f"rollout_ep{ep+1}.gif")
                imageio.mimsave(fn, frames, duration=0.08)
                print(f"saved {fn}")
            except Exception as e:
                print("recording skipped (install imageio). Error:", e)

    np.savez(os.path.join(args.outdir, "summary.npz"),
             returns=np.array(returns, dtype=np.float32),
             lengths=np.array(lengths, dtype=np.int32),
             seed=args.seed)
    print("avg_return:", float(np.mean(returns)), "avg_len:", float(np.mean(lengths)))

if __name__ == "__main__":
    main()
