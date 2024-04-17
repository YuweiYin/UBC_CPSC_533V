import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch

from util.agents.sac_agent import SACAgent
from util.infrastructure.rl_trainer import RLTrainer


class SACTrainer(object):

    def __init__(self, params):
        # Soft Actor-Critic

        computation_graph_args = {
            "n_layers": params["n_layers"],
            "size": params["size"],
            "learning_rate": params["learning_rate"],
            "init_temperature": params["init_temperature"],
            "actor_update_frequency": params["actor_update_frequency"],
            "critic_target_update_frequency": params["critic_target_update_frequency"]
        }
        estimate_advantage_args = {
            "gamma": params["discount"],
        }
        train_args = {
            "num_agent_train_steps_per_iter": params["num_agent_train_steps_per_iter"],
            "num_critic_updates_per_agent_update": params["num_critic_updates_per_agent_update"],
            "num_actor_updates_per_agent_update": params["num_actor_updates_per_agent_update"],
        }

        self.agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.actor_ckpt = params["actor_ckpt"]
        self.critic_ckpt = params["critic_ckpt"]

        self.data_dir = params["data_dir"]
        self.max_n_traj = params["max_n_traj"]
        self.max_traj_len = params["max_traj_len"]

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        self.params = params
        self.params["agent_class"] = SACAgent
        self.params["agent_params"] = self.agent_params
        self.params["batch_size_initial"] = self.params["batch_size"]

        self.rl_trainer = RLTrainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_sac_training_loop(
            self.params["n_iter"],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )

    def run_collect_dataset(
            self,
            level: str = "random",
    ):
        self.rl_trainer.collect_offline_dataset(
            save_dir=self.data_dir,
            env_name=self.params["env_name"].split("-")[0],
            env_version=self.params["env_name"].split("-")[-1],
            level=level,
            max_n_traj=self.max_n_traj,
            max_traj_len=self.max_traj_len,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose mode: show logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of all modules")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device(s), e.g., 0 OR 0,1")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Use GPU or not")

    parser.add_argument("-a", "--actor_ckpt", type=str, default="", help="The checkpoint path of Actor model")
    parser.add_argument("-c", "--critic_ckpt", type=str, default="", help="The checkpoint path of Critic model")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="The directory to save checkpoints")
    parser.add_argument("--load_ckpt", action="store_true", default=False)
    parser.add_argument("--save_params", action="store_true", default=False)

    parser.add_argument("--data_dir", type=str, default="data", help="The directory to save datasets")
    parser.add_argument("--level", type=str, default="ALL",
                        help="The dataset level to collect: \"random\", \"medium\", \"expert\", \"ALL\".")
    parser.add_argument("--max_n_traj", type=int, default=1000000, help="Max number of collected trajectories.")
    parser.add_argument("--max_traj_len", type=int, default=150, help="Max length of each collected trajectory.")

    # parser.add_argument("--env_name", type=str, default="CartPole-v0")
    parser.add_argument(
        "--env_name",
        default="HalfCheetah-v4",
        choices=("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")
    )
    parser.add_argument("--ep_len", type=int, default=200)
    parser.add_argument("--exp_name", type=str, default="SAC_Collecting_Datasets")
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--num_agent_train_steps_per_iter", type=int, default=1)
    parser.add_argument("--num_critic_updates_per_agent_update", type=int, default=1)
    parser.add_argument("--num_actor_updates_per_agent_update", type=int, default=1)
    parser.add_argument("--actor_update_frequency", type=int, default=1)
    parser.add_argument("--critic_target_update_frequency", type=int, default=1)
    parser.add_argument("--batch_size", "-b", type=int, default=1000)  # steps collected per train iteration
    parser.add_argument("--eval_batch_size", "-eb", type=int, default=400)  # steps collected per eval iteration
    parser.add_argument("--train_batch_size", "-tb", type=int, default=256)  # steps used per gradient step

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--init_temperature", "-temp", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=64)

    parser.add_argument("--no_gpu", "-ngpu", action="store_true", default=False)
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=10)

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

    # Set the random seed of all modules
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.has_cuda:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set logging directory
    output_dir = os.path.join("log", "collect_offline_data")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logdir = f"{args.exp_name}_{args.env_name}_" + time.strftime("%Y-%m-%d_%H:%M:%S")
    logdir = os.path.join(output_dir, logdir)
    params["logdir"] = logdir
    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    # Run collecting
    trainer = SACTrainer(params)
    # trainer.run_training_loop()
    match args.level:
        case "random":
            trainer.run_collect_dataset(level="random")
        case "medium":
            trainer.run_collect_dataset(level="medium")
        case "expert":
            trainer.run_collect_dataset(level="expert")
        case "ALL":
            trainer.run_collect_dataset(level="random")
            trainer.run_collect_dataset(level="medium")
            trainer.run_collect_dataset(level="expert")
        case _:
            raise ValueError(f"ValueError: level = {args.level}")


if __name__ == "__main__":
    timer_start = time.perf_counter()

    main()

    timer_end = time.perf_counter()
    logging.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
