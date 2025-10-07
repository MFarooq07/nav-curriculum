# Curriculum Navigation: Rays → Pixels

Tiny RL project to show that a simple curriculum (ray distances → tiny images) improves sample efficiency and final success for maze navigation.
CPU-only, Python-only. Matrix: rays-baseline, pixels-from-scratch, pixels-with-curriculum.

## Quickstart
1) Create venv and install deps:
python -m venv .venv
..venv\Scripts\Activate.ps1
pip install -r requirements.txt
2) (Coming next days) `python train.py --obs rays|pixels|curriculum`
