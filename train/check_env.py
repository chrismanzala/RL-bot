"""Quick sanity check that the learning stack is installed and VMAS runs.

    python train/check_env.py                 # uses the GPU if there is one
    python train/check_env.py --device cuda   # fail unless the GPU is usable
"""
import argparse

import torch
import vmas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    device = p.parse_args().device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"torch {torch.__version__} | vmas {vmas.__version__}")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        print(f"CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")

    env = vmas.make_env("navigation", num_envs=4, n_agents=3, device=device, seed=0)
    obs = env.reset()
    print(f"{len(obs)} agents, obs shape per agent: {tuple(obs[0].shape)}")
    for _ in range(10):
        actions = [env.get_random_action(agent) for agent in env.agents]
        obs, rews, dones, _ = env.step(actions)
    print(f"Stepped 10 random actions OK on {device}, mean reward:", torch.stack(rews).mean().item())


if __name__ == "__main__":
    main()
