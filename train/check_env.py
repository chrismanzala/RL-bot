"""Quick sanity check that the learning stack is installed and VMAS runs."""
import torch
import vmas


def main() -> None:
    print(f"torch {torch.__version__} | vmas {vmas.__version__}")
    env = vmas.make_env("navigation", num_envs=4, n_agents=3, device="cpu", seed=0)
    obs = env.reset()
    print(f"{len(obs)} agents, obs shape per agent: {tuple(obs[0].shape)}")
    for _ in range(10):
        actions = [env.get_random_action(agent) for agent in env.agents]
        obs, rews, dones, _ = env.step(actions)
    print("Stepped 10 random actions OK, mean reward:", torch.stack(rews).mean().item())


if __name__ == "__main__":
    main()
