from envs.grid_nav import GridNav, GridConfig
from envs.wrappers import make_gridnav

def test_step_and_done():
    env, spec = make_gridnav(GridConfig(seed=999))
    obs = env.reset()
    assert obs.shape == spec.observation_space.shape
    done = False
    steps = 0
    while not done and steps < 50:
        a = spec.action_space.sample()
        obs, r, done, info = env.step(a)
        steps += 1
    assert steps > 0
