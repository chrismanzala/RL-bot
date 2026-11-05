"""Train a multi-robot policy with MAPPO (or IPPO) on a VMAS scenario.

Uses centralized training / decentralized execution: during training the
critic may see the whole team, but each robot's policy acts only on its own
observation, so the learned policy can run on each robot independently.

Examples:
    python train/train_mappo.py                       # MAPPO, 3 robots, navigation
    python train/train_mappo.py --algo ippo           # independent PPO baseline
    python train/train_mappo.py --n-agents 5 --frames 2000000
    python train/train_mappo.py --quick               # ~10 second smoke test
    python train/train_mappo.py --no-render           # on a machine without a display
"""
import argparse
from pathlib import Path

from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

ALGOS = {"mappo": MappoConfig, "ippo": IppoConfig}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, default="mappo")
    p.add_argument("--task", default="navigation", help="VMAS task name, e.g. navigation, balance")
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--frames", type=int, default=1_000_000, help="total environment steps to train for")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
    p.add_argument("--no-render", action="store_true", help="skip evaluation videos (needed on headless servers)")
    p.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    task = getattr(VmasTask, args.task.upper()).get_from_yaml()
    if "n_agents" in task.config:
        task.config["n_agents"] = args.n_agents

    exp = ExperimentConfig.get_from_yaml()
    exp.sampling_device = exp.train_device = args.device
    exp.buffer_device = args.device
    exp.max_n_frames = args.frames
    exp.loggers = ["csv", "wandb"] if args.wandb else ["csv"]
    exp.save_folder = str(Path("results").resolve())
    Path(exp.save_folder).mkdir(exist_ok=True)
    exp.checkpoint_at_end = True
    exp.render = not args.no_render

    if args.quick:
        exp.max_n_frames = 12_000
        exp.on_policy_collected_frames_per_batch = 6_000
        exp.on_policy_n_minibatch_iters = 2
        exp.evaluation_interval = 6_000
        exp.evaluation_episodes = 2
        exp.render = False
    else:
        exp.evaluation_interval = 60_000

    experiment = Experiment(
        task=task,
        algorithm_config=ALGOS[args.algo].get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=exp,
    )
    experiment.run()
    print(f"\nDone. Logs, videos and checkpoint are in {experiment.folder_name}")


if __name__ == "__main__":
    main()
