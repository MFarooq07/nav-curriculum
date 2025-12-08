## Curriculum Navigation: Rays Pixels- A 10 day PROJECT!!!
 
Tiny RL project to show that a simple curriculum (ray distances → tiny images) improves sample efficiency and final success for maze navigation.
CPU-only, Python-only. Matrix: rays-baseline, pixels-from-scratch, pixels-with-curriculum.

## Day 1: Project bootstrap

* Make repo folders & requirements.txt; python -m venv .venv && pip install -r requirements.txt.

* Write `utils/log.py` with:` set_seed(seed)`, simple logger, and `save_video(frames, path)`.

* Create `README.md` with a one-paragraph goal + “How to run” placeholder or whatever you like.

* Finish line: env set up, seed util done.


## Day 2: Minimal grid world (ray sensors)
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

 ## Day 3: CLI runner + rollouts + CSV logs + plotting
* Make sure you are in the correct directory `nav-curriculum` (or whatever you have named it) and have activated the virtual environment
* Instal if you haven't already the following
    * numpy
    * pandas
    * matplotlib
as follows: `pip install numpy pandas matplotlib`   
* Now create `cli.py`
* Create `plots.py` to make the plots of the results
* Run rollouts

`python .\cli.py --episodes 200 --seed 123 --out .\resultsl`

* and make plots 

`python .\plots.py --csv (Get-ChildItem .\results\run_*\metrics.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName`


# A RECAP OF WHAT WE HAVE DONE SO FAR TILL DAY 3?
* **Made a 2d maze with random walls:** there's an agent (red) and a goal (green). the agent i.e. robot can move up, down, left, right.if it reaches wall then collision; if border end and if goal thats a success

* **Observations** the agent “sees” with 8 short rays around it (how far to the nearest wall in each direction) + a unit vector to the goal (goal direction).

* **Rewards & Endings** small step penalty each move, big reward on goal, penalty + terminate on collision, or stop at a max step limit.

* **CLI runner + Logging** a command-line script runs many episodes with a random policy, logs per-episode results (return, steps, success, etc.) to CSV.

* **a success plot** we load the CSV and plot success rate over time to see whether episodes are reaching the goal at all with the current (random) policy.

 ## Day 4: Classical baseline (A* follower) vs Random
 **Goal**: add a simple classical planner (A*) and benchmark it against a random policy on the Day-2 grid world. Log results to CSV and plot success trends.

**Files to be added/changed:**
* `envs/policies.py` contains:
    * `AStarFollower.plan(grid, start, goal) -> list[path cells]`
    * `AStarFollower.next_action(agent_pos) -> {0,1,2,3}` (maps path steps to UP/DOWN/LEFT/RIGHT)
    * `cli.py` now accepts `--policy {random, astar}` and writes episode logs:
        * CSV columns: `episodes, success, steps, event`
    * `analyze_day4.py` reads the two CSVs and prints summary stats + saves moving-average success plots

    ### Smoke test (1-5 episodes)
    **Powershell commands**
    `mkdir -Force.\results` 

    * run the experiment with `random policy`

    `python .\cli.py --policy random --episodes 5 --seed 123 --out .\results\smoke_random.csv`

    * run another with `A*` policy 

    `python .\cli.py --policy astar  --episodes 5 --seed 123 --out .\results\smoke_astar.csv`

    Open the CSVs at their location to confirm rows like: `1, 0, 3, collision` etc.

    Full benchmark (same env settings for fairness) with 1000 espisodes/policy

    `# Random — 1000 episodes`
    
    `python .\cli.py --policy random --episodes 1000 --seed 123 --wall_prob 0.18 --H 15 --W 15 --ray_max 10 --out .\results\day4_random.csv`

    `# A* — 1000 episodes`

    `python .\cli.py --policy astar  --episodes 1000 --seed 123 --wall_prob 0.18 --H 15 --W 15 --ray_max 10 --out .\results\day4_astar.csv`

    * To visualize the analysis, plot using the follwoing powershell commands

    python .\analyze_day4.py `
    
    --random_csv .\results\day4_random.csv `

    --astar_csv  .\results\day4_astar.csv `

    --out_dir    .\results `

    --ma 20`


**Outputs you should be seeing**
* Console stats (example patterns):
    * Random: low success rate (0-5%), many collisions
    * A*: very high success rate (~95-99%), few collisions 
* Plots in `results/`:
    * `day4_random_success_ma.png`
    * `day4_astar_success_ma.png`
    (Moving average success (MA=20) showing A* >> Random- as anticipated)

**Finish line checks**
* A* file import works:
`python -c "from envs.policies import AStarFollower; print('A* OK')"`
* CLI help shows the new flag:
`python .\cli.py -h` should list: --policy {random,astar}
**Notes**
* If you get "Permission denied" for CSVs, make sure the file isn't open elsewhere and that `results\ exists.
* Keep the same -seed, map size, and `wall_prob` across policies to make comparisons fair.
* If you change grid/action conventions, ensrure `ACTIONS` in `envs/grid_nav.py` matches what AStarFollower emits.