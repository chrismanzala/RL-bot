# RL-bot

Decentralized multi-robot swarm trained with multi-agent reinforcement learning (MARL).

Each robot runs its own policy on local observations. Policies are trained together in
simulation with centralized training / decentralized execution (CTDE), then moved to
ROS 2 robots. See the full project overview for objectives, scope and success criteria.

## Status

Early stage. What works today:

- `train/train_mappo.py` trains MAPPO or IPPO on a VMAS multi-robot scenario (default:
  3 robots navigating to goals without colliding). On CPU, 240k steps take a few minutes
  and the mean episode return rises from about -0.4 to about 0.95.
- `ROS.cpp` is a minimal ROS 2 (`rclcpp`) node that drives turtlesim. It is the starting
  point for running trained policies on robots. It has not yet been built in CI.
- `bag_files/` holds ROS 2 turtlesim recordings.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train/check_env.py               # checks torch + VMAS work
python train/train_mappo.py --quick     # ~10 second smoke test
python train/train_mappo.py             # full run: MAPPO, 3 robots, 1M steps
```

Useful options:

| Option | What it does |
|---|---|
| `--algo ippo` | Independent PPO baseline instead of MAPPO |
| `--n-agents 5` | Number of robots |
| `--frames 2000000` | Training length in environment steps |
| `--task balance` | Another VMAS task |
| `--device cuda` | Train on a GPU |
| `--no-render` | Skip evaluation videos (needed on machines without a display) |
| `--wandb` | Also log to Weights & Biases |

Results (CSV metrics, videos and a checkpoint) are written to `results/`.

## Layout

```
train/       training and sanity-check scripts
envs/        custom swarm scenarios (next: formation keeping)
ROS.cpp      ROS 2 node, to become the policy runner on robots
bag_files/   ROS 2 recordings
```

## Next steps

1. Write a formation-keeping VMAS scenario in `envs/` and compare MAPPO with IPPO.
2. Test robustness: remove robots mid-episode, add sensor noise and message loss.
3. Load a trained policy in a ROS 2 node that drives several turtlesim turtles.
