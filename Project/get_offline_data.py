import os
import sys
import time
import pickle
import random
import logging
import argparse
import collections

import numpy as np
# import requests
import h5py
# import d4rl
from data.data_infos import DATASET_URLS
# import gym
# import gymnasium as gym
from transformers import set_seed


def download_offline_data():
    # datasets = [
    #     "halfcheetah-random-v0", "halfcheetah-medium-v0", "halfcheetah-expert-v0",
    #     "halfcheetah-medium-replay-v0", "halfcheetah-medium-expert-v0",
    #     "walker2d-random-v0", "walker2d-medium-v0", "walker2d-expert-v0",
    #     "walker2d-medium-replay-v0", "walker2d-medium-expert-v0",
    #     "hopper-random-v0", "hopper-medium-v0", "hopper-expert-v0",
    #     "hopper-medium-replay-v0", "hopper-medium-expert-v0",
    #     "ant-random-v0", "ant-medium-v0", "ant-expert-v0",
    #     "ant-medium-replay-v0", "ant-medium-expert-v0",
    # ]  # "ant-random-expert-v0"

    datasets = []
    assert isinstance(DATASET_URLS, dict)
    for ds_name in args.datasets:
        for ds_level in args.levels:
            cur_ds = f"{ds_name}-{ds_level}-{args.version}"
            if cur_ds in DATASET_URLS and isinstance(DATASET_URLS[cur_ds], str):  # has url
                datasets.append(cur_ds)
            else:
                logger.info(f">>> Key Error: dataset `{cur_ds}` not found")
    logger.info(f">>> There are {len(datasets)} offline datasets to download.")

    save_raw_dir = os.path.join(args.save_dir, "raw")
    if not os.path.isdir(save_raw_dir):
        os.makedirs(save_raw_dir, exist_ok=True)

    for cur_ds in datasets:
        assert cur_ds in DATASET_URLS
        url = DATASET_URLS[cur_ds]
        target_fp = os.path.join(save_raw_dir, f"{cur_ds}.hdf5")
        if os.path.isfile(target_fp):  # the file exists
            logger.info(f">>> >>> Skip {target_fp}")
            continue
        # todo_command = f"wget -P {save_raw_dir}/ {url}"
        todo_command = f"wget --no-verbose -O {target_fp} {url}"  # --quiet
        logger.info(f">>> >>> $ {todo_command}")
        try:
            os.system(todo_command)  # run wget downloading (macOS: `brew install wget`)
        except Exception as e:
            logger.info(f"Exception in `{todo_command}`\n{e}")


def parse_offline_data():
    save_raw_dir = os.path.join(args.save_dir, "raw")
    assert os.path.isdir(save_raw_dir)

    ds_filenames = os.listdir(save_raw_dir)
    ds_filenames = [ds_fn for ds_fn in ds_filenames if ds_fn.endswith(".hdf5")]
    ds_fn_set = set(ds_filenames)

    datasets = []
    ds_dict = dict()
    assert isinstance(DATASET_URLS, dict)
    for ds_name in args.datasets:
        ds_dict[ds_name] = dict()

        # Set the env_name for Gymnasium
        # Although the offline dataset is from v2 env, we use v4 env for gym.make to avoid using mujoco-py (buggy)
        # https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/mujoco_py_env.py#L15
        if ds_name == "halfcheetah":
            ds_dict[ds_name]["env_name"] = f"HalfCheetah-v4"  # f"HalfCheetah-{args.version}"
        elif ds_name == "walker2d":
            ds_dict[ds_name]["env_name"] = f"Walker2d-v4"  # f"Walker2d-{args.version}"
        elif ds_name == "hopper":
            ds_dict[ds_name]["env_name"] = f"Hopper-v4"  # f"Hopper-{args.version}"
        elif ds_name == "ant":
            ds_dict[ds_name]["env_name"] = f"Ant-v4"  # f"Ant-{args.version}"
        else:
            raise ValueError(f"ValueError: ds_name = {ds_name}")

        ds_dict[ds_name]["levels"] = []
        for ds_level in args.levels:
            cur_ds = f"{ds_name}-{ds_level}-{args.version}.hdf5"
            if cur_ds in ds_fn_set:  # exist
                datasets.append(cur_ds)
                ds_dict[ds_name][ds_level] = cur_ds  # level values (filenames)
                ds_dict[ds_name]["levels"].append(ds_level)  # level keys
            else:
                logger.info(f">>> Key Error: dataset `{cur_ds}` not found")
    logger.info(f">>> There are {len(datasets)} offline datasets to parse.")

    save_parsed_dir = os.path.join(args.save_dir, "parsed")
    if not os.path.isdir(save_parsed_dir):
        os.makedirs(save_parsed_dir, exist_ok=True)

    show_log = 100
    for ds_name, ds_info in ds_dict.items():
        # env_name = ds_info["env_name"]
        ds_levels = ds_info["levels"]
        ds_fn_list = [ds_info[level] for level in ds_levels]

        for ds_fn in ds_fn_list:
            ds_fp = os.path.join(save_raw_dir, ds_fn)
            with h5py.File(ds_fp, "r") as fp_in:
                logger.info(f">>> Dataset: {ds_fn}")
                cur_keys = fp_in.keys()
                logger.info(f">>> Keys: {cur_keys}")
                # <KeysViewHDF5 ["actions", "infos", "next_observations",
                #     "observations", "rewards", "terminals", "timeouts"]>

                # infos = fp_in.get("infos")  # <HDF5 group "/infos" (3 members)>
                actions = fp_in.get("actions")  # <HDF5 dataset: shape (n_samples, n_actions), type "<f4">
                obs = fp_in.get("observations")  # <HDF5 dataset: shape (n_samples, n_obs), type "<f4">
                next_obs = fp_in.get("next_observations")  # <HDF5 dataset: shape (n_samples, n_obs), type "<f4">
                rewards = fp_in.get("rewards")  # <HDF5 dataset: shape (n_samples,), type "<f4">
                terminals = fp_in.get("terminals")  # <HDF5 dataset: shape (n_samples,), type "|b1">
                timeouts = fp_in.get("timeouts")  # <HDF5 dataset: shape (n_samples,), type "|b1">

                # logger.info(type(actions[0]))  # <class "numpy.ndarray">
                # logger.info(type(actions[0][0]))  # <class "numpy.float32">
                # logger.info(type(obs[0]))  # <class "numpy.ndarray">
                # logger.info(type(obs[0][0]))  # <class "numpy.float32">
                # logger.info(type(next_obs[0]))  # <class "numpy.ndarray">
                # logger.info(type(next_obs[0][0]))  # <class "numpy.float32">
                # logger.info(type(rewards[0]))  # <class "numpy.float32">
                # logger.info(type(terminals[0]))  # <class "numpy.bool_">
                # logger.info(type(timeouts[0]))  # <class "numpy.bool_">

                # env = gym.make(env_name, render_mode=None)
                # # env = gym.make(env_name, render_mode="human")
                # obs, info = env.reset(seed=args.seed)  # get first obs/state; set random seed for the env
                # # dataset = env.get_dataset()  # d4rl get_dataset (https://github.com/Farama-Foundation/D4RL)

                n_samples = rewards.shape[0]
                use_timeouts = timeouts is not None

                episode_step = 0
                trajectories = []
                n_tj = 0
                trajectory_chunk = collections.defaultdict(list)
                for i in range(n_samples):
                    done_bool = bool(terminals[i])
                    if use_timeouts:
                        final_timestep = bool(timeouts[i])
                    else:
                        final_timestep = (episode_step + 1) == args.max_len_trajectory

                    trajectory_chunk["obs"].append(obs[i])
                    trajectory_chunk["next_obs"].append(next_obs[i])
                    trajectory_chunk["actions"].append(actions[i])
                    trajectory_chunk["rewards"].append(rewards[i])
                    trajectory_chunk["terminals"].append(terminals[i])

                    if done_bool or final_timestep:  # finish collecting a chunk of the whole trajectory
                        cur_trajectory = dict()
                        for k in trajectory_chunk:
                            cur_trajectory[k] = np.array(trajectory_chunk[k])
                        trajectories.append(cur_trajectory)  # add this trajectory
                        n_tj += 1
                        if n_tj % show_log == 0:
                            logger.info(f">>> >>> Parsing >>> {ds_fn}: {n_tj} trajectories")

                        episode_step = 0  # reset
                        trajectory_chunk = collections.defaultdict(list)  # reset

                    episode_step += 1

                # Show statistics
                tj_rewards = np.array([np.sum(tj["rewards"]) for tj in trajectories])
                n_samples = np.sum([tj["rewards"].shape[0] for tj in trajectories])
                logger.info(f"The number of collected samples: {n_samples}")
                logger.info(f"Trajectory returns: mean = {np.mean(tj_rewards)}, std = {np.std(tj_rewards)}, "
                            f"max = {np.max(tj_rewards)}, min = {np.min(tj_rewards)}")

                save_parsed_fp = os.path.join(save_parsed_dir, f"{ds_fn}.pkl")
                logger.info(f">>> Done {ds_fn}. Save the trajectories to {save_parsed_fp}")
                with open(save_parsed_fp, "wb") as f_out:
                    pickle.dump(trajectories, f_out)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose mode: show logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of all modules")
    parser.add_argument("--save_dir", type=str, default="data", help="The directory to save data")
    parser.add_argument("--task", type=str, default="all",
                        help="Data task: \"all\", \"download\", \"parse\"")
    parser.add_argument("--data", type=str, default="all",
                        help="Dataset name: \"all\", \"halfcheetah\", \"walker2d\", \"hopper\", \"ant\"")
    parser.add_argument("--level", type=str, default="all",
                        help="Dataset name: \"all\", \"random\", \"medium\", \"expert\", "
                             "\"medium-replay\", \"medium-expert\"")
    parser.add_argument("--version", type=str, default="v2",
                        help="Offline data version: \"v2\", \"v1\", \"v0\"")
    parser.add_argument("-m", "--max_len_trajectory", type=int, default=1000,
                        help="The max length of each trajectory (offline data sample)")

    args = parser.parse_args()
    logger.info(args)
    args.logger = logger

    timer_start = time.perf_counter()

    # Set the random seed of all modules
    args.seed = int(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    # The max length of each trajectory (offline data sample)
    args.max_len_trajectory = max(int(args.max_len_trajectory), 1000)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.data == "all":
        args.datasets = ["halfcheetah", "hopper", "walker2d", "ant"]
    elif isinstance(args.data, str):
        args.datasets = [args.data]
    else:
        raise ValueError(f"ValueError: args.data = {args.data}")

    if args.level == "all":
        args.levels = ["random", "medium", "expert", "medium-replay", "medium-expert"]
    elif isinstance(args.level, str):
        args.levels = [args.level]
    else:
        raise ValueError(f"ValueError: args.level = {args.level}")

    if args.task == "all":
        download_offline_data()
        parse_offline_data()
    elif args.task == "download":
        download_offline_data()
    elif args.task == "parse":
        parse_offline_data()
    else:
        raise ValueError(f"ValueError: args.task = {args.task}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
