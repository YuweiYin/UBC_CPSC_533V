import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch

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

    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose mode: show logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of all modules")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device(s), e.g., 0 OR 0,1")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Use GPU or not")

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

    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--scalar_log_freq", type=int, default=int(1e4))
    parser.add_argument("--video_log_freq", type=int, default=-1)

    parser.add_argument("--save_params", action="store_true")

    args = parser.parse_args()

    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.has_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.has_cuda and args.use_gpu else "cpu")
    args.gpus = args.cuda.split(",") if "," in args.cuda else [args.cuda]
    args.gpus = [int(gpu_id) for gpu_id in args.gpus]
    args.device_count = int(torch.cuda.device_count())
    args.ddp_able = args.has_cuda and len(args.gpus) > 1 and args.device_count > 1
    if args.verbose:
        logging.info(
            f"HAS_CUDA: {args.has_cuda}; DEVICE: {args.device}; GPUS: {args.gpus}; DDP able: {args.ddp_able}")
        logging.info(f"torch.__version__: {torch.__version__}")
        logging.info(f"torch.version.cuda: {torch.version.cuda}")
        logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logging.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        logging.info(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
        logging.info(f"torch.cuda.get_arch_list(): {torch.cuda.get_arch_list()}")
        if args.has_cuda:
            logging.info(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
            logging.info(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

    params = vars(args)  # convert to dictionary
    params["video_log_freq"] = -1  # This param is not used for DQN

    # Set the random seed of all modules
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.has_cuda:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set logging directory
    output_dir = os.path.join("log", "train_dqn")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logdir = f"{args.exp_name}_{args.env_name}_" + time.strftime("%Y-%m-%d_%H:%M:%S")
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
