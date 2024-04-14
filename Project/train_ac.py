import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch

from util.agents.ac_agent import ACAgent
from util.infrastructure.rl_trainer import RLTrainer


class ACTrainer(object):

    def __init__(self, params):
        # Actor-Critic

        computation_graph_args = {
            "n_layers": params["n_layers"],
            "size": params["size"],
            "learning_rate": params["learning_rate"],
            "num_target_updates": params["num_target_updates"],
            "num_grad_steps_per_target_update": params["num_grad_steps_per_target_update"],
        }
        estimate_advantage_args = {
            "gamma": params["discount"],
            "standardize_advantages": not (params["dont_standardize_advantages"]),
        }
        train_args = {
            "num_agent_train_steps_per_iter": params["num_agent_train_steps_per_iter"],
            "num_critic_updates_per_agent_update": params["num_critic_updates_per_agent_update"],
            "num_actor_updates_per_agent_update": params["num_actor_updates_per_agent_update"],
        }

        self.agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params["agent_class"] = ACAgent
        self.params["agent_params"] = self.agent_params
        self.params["batch_size_initial"] = self.params["batch_size"]

        self.rl_trainer = RLTrainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.params["n_iter"],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose mode: show logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of all modules")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device(s), e.g., 0 OR 0,1")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Use GPU or not")

    # parser.add_argument("--env_name", type=str, default="CartPole-v0")
    parser.add_argument(
        "--env_name",
        default="HalfCheetah-v4",
        choices=("HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "Ant-v4")
    )
    parser.add_argument("--ep_len", type=int, default=200)
    parser.add_argument("--exp_name", type=str, default="AC_Training")
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--num_agent_train_steps_per_iter", type=int, default=1)
    parser.add_argument("--num_critic_updates_per_agent_update", type=int, default=1)
    parser.add_argument("--num_actor_updates_per_agent_update", type=int, default=1)

    parser.add_argument("--batch_size", "-b", type=int, default=1000)  # steps collected per train iteration
    parser.add_argument("--eval_batch_size", "-eb", type=int, default=400)  # steps collected per eval iteration
    parser.add_argument("--train_batch_size", "-tb", type=int, default=1000)  # steps used per gradient step

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--dont_standardize_advantages", "-dsa", action="store_true")
    parser.add_argument("--num_target_updates", "-ntu", type=int, default=10)
    parser.add_argument("--num_grad_steps_per_target_update", "-ngsptu", type=int, default=10)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=64)

    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=10)

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

    # For policy gradient, we made a design decision to force batch_size = train_batch_size.
    # Note that, to avoid confusion, you do not even have a train_batch_size argument anymore (above)
    params["train_batch_size"] = params["batch_size"]

    # Set the random seed of all modules
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.has_cuda:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set logging directory
    output_dir = os.path.join("log", "train_ac")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logdir = f"{args.exp_name}_{args.env_name}_" + time.strftime("%Y-%m-%d_%H:%M:%S")
    logdir = os.path.join(output_dir, logdir)
    params["logdir"] = logdir
    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    # Run training
    trainer = ACTrainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    timer_start = time.perf_counter()

    main()

    timer_end = time.perf_counter()
    logging.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)