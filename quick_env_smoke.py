from envs.grid_nav import GridNav, GridConfig
import imageio.v2 as iio
try:
    env = GridNav(GridConfig())
    env.reset(seed=7)
    img = env.render_rgb(scale=16)
    iio.imwrite("results\\day2_smoke.png", img)
    print("wrote results/day2_smoke.png")
except Exception as e:
    print("render test failed:", e)
