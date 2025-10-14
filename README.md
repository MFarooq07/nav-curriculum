## Curriculum Navigation: Rays Pixels- A 10 day PROJECT!!!
 
Tiny RL project to show that a simple curriculum (ray distances → tiny images) improves sample efficiency and final success for maze navigation.
CPU-only, Python-only. Matrix: rays-baseline, pixels-from-scratch, pixels-with-curriculum.

## Day 1 Project bootstrap

* Make repo folders & requirements.txt; python -m venv .venv && pip install -r requirements.txt.

* Write `utils/log.py` with:` set_seed(seed)`, simple logger, and `save_video(frames, path)`.

* Create `README.md` with a one-paragraph goal + “How to run” placeholder or whatever you like.

* Finish line: env set up, seed util done.


## Day 2 Minimal grid world (ray sensors)
* In `envs/grid_nav.py`: 2D grid with walls, start/goal; 4 actions (↑↓←→) or left/right/forward if you prefer diff-drive.

* Observation = k=8 ray distances (normalized 0..1) + goal direction (dx,dy) clipped.

* `reset()` returns obs; `step(a)` returns (obs, reward, done, info).

Reward: +1 on goal, -0.01 step cost, -1 on collision, terminate on goal/collision/200 steps.

Finish line: `python -c "from envs.grid_nav import GridNav; print(GridNav().reset())"` works.

To check if the day 2 has been correctly implemented and is working fine, open the terminal in Visual Studio (keyboard shortcut `ctrl + shift + '`) or the Windows Poweshell and run the following commands.
`@'`
`from envs.grid_nav import GridNav, GridConfig`
`env = GridNav(GridConfig(seed=123))`
`obs = env.reset()`
`total = 0.0`
`for t in range(20):`
`    obs, r, done, info = env.step(0)`
`    total += r`
`    if done:`
`        break`
`print("steps:", t+1, "return:", round(total,3), "done:", done, "info:", info)`
`print("obs_len:", len(obs))`
`'@ | python -`

 





