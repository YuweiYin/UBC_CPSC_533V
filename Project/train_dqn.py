import os
import sys
import time
import logging
import argparse

from util.agents.dqn_agent import DQNAgent
from util.infrastructure.rl_trainer import RLTrainer
from util.infrastructure.dqn_utils import get_env_kwargs


class QTrainer(object):

    def __init__(self, params):
        # Deep Q-learning

        train_args = {
            "num_agent_train_steps_per_iter": params["num_agent_train_steps_per_iter"],
            "num_critic_updates_per_agent_update": params["num_critic_updates_per_agent_update"],
            "train_batch_size": params["batch_size"],
            "double_q": params["double_q"],
        }
        env_args = get_env_kwargs(params["env_name"])

        self.agent_params = {**train_args, **env_args, **params}

        self.params = params
        self.params["agent_class"] = DQNAgent
        self.params["agent_params"] = self.agent_params
        self.params["train_batch_size"] = params["batch_size"]
        self.params["env_wrappers"] = self.agent_params["env_wrappers"]

        self.rl_trainer = RLTrainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params["num_timesteps"],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env_name",
        default="LunarLander-v3",
        choices=("PongNoFrameskip-v4", "LunarLander-v3", "MsPacman-v0")
    )
    # parser.add_argument(
    #     "--env_name",
    #     default="HalfCheetah-v4",
    #     choices=("HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "Ant-v4")
    # )

    parser.add_argument("--ep_len", type=int, default=200)
    parser.add_argument("--exp_name", type=str, default="todo")

    parser.add_argument("--eval_batch_size", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_agent_train_steps_per_iter", type=int, default=1)
    parser.add_argument("--num_critic_updates_per_agent_update", type=int, default=1)
    parser.add_argument("--double_q", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--scalar_log_freq", type=int, default=int(1e4))
    parser.add_argument("--video_log_freq", type=int, default=-1)

    parser.add_argument("--save_params", action="store_true")

    args = parser.parse_args()
    params = vars(args)  # convert to dictionary
    params["video_log_freq"] = -1  # This param is not used for DQN

    # Set logging directory
    output_dir = os.path.join("log", "train_dqn")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logdir = f"{args.exp_name}_{args.env_name}_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(output_dir, logdir)
    params["logdir"] = logdir
    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    # Run training
    trainer = QTrainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    timer_start = time.perf_counter()

    main()

    timer_end = time.perf_counter()
    logging.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
