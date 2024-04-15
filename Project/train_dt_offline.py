import os
import sys
import time
import random
import pickle
import logging
import argparse

import numpy as np
import torch
# import gym
import gymnasium as gym
from transformers import set_seed
import wandb

from data.data_infos import DATASET_URLS
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_dt
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def get_dataset_dict(
        target_env: str = ""
) -> dict:
    save_parsed_dir = os.path.join(args.save_dir, "parsed")
    assert os.path.isdir(save_parsed_dir)

    ds_filenames = os.listdir(save_parsed_dir)
    ds_filenames = [ds_fn for ds_fn in ds_filenames if ds_fn.endswith(".hdf5.pkl")]
    ds_fn_set = set(ds_filenames)

    datasets = []
    ds_dict = dict()
    assert isinstance(DATASET_URLS, dict)
    for env_name in args.env_list:
        if len(target_env) > 0 and env_name != target_env:
            continue

        ds_dict[env_name] = dict()

        # Set the env_name for Gymnasium
        # Although the offline dataset is from v2 env, we use v4 env for gym.make to avoid using mujoco-py (buggy)
        # https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/mujoco_py_env.py#L15
        if env_name == "halfcheetah":
            ds_dict[env_name]["gymnasium_env"] = f"HalfCheetah-v4"  # f"HalfCheetah-{args.version}"
        elif env_name == "walker2d":
            ds_dict[env_name]["gymnasium_env"] = f"Walker2d-v4"  # f"Walker2d-{args.version}"
        elif env_name == "hopper":
            ds_dict[env_name]["gymnasium_env"] = f"Hopper-v4"  # f"Hopper-{args.version}"
        # elif env_name == "ant":
        #     ds_dict[env_name]["gymnasium_env"] = f"Ant-v4"  # f"Ant-{args.version}"
        else:
            raise ValueError(f"ValueError: env_name = {env_name}")

        ds_dict[env_name]["levels"] = []
        for level in args.levels:
            ds_name = f"{env_name}-{level}-{args.version}.hdf5.pkl"
            if ds_name in ds_fn_set:  # exist
                datasets.append(ds_name)
                ds_dict[env_name][level] = ds_name  # level values (filenames)
                ds_dict[env_name]["levels"].append(level)  # level keys
            else:
                logger.info(f">>> Key Error: dataset `{ds_name}` not found")

    if len(target_env) > 0:
        logger.info(f">>> There are {len(datasets)} parsed offline datasets for env {target_env}.")
    else:
        logger.info(f">>> There are {len(datasets)} parsed offline datasets")

    return ds_dict


def discount_cumsum(x, gamma):
    discount_cum_sum = np.zeros_like(x)
    discount_cum_sum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cum_sum[t] = x[t] + gamma * discount_cum_sum[t+1]
    return discount_cum_sum


def run(
        exp_prefix: str = "gym-mujoco",
):
    device = args.device
    seed = int(args.seed)
    log_to_wandb = bool(args.log_to_wandb)

    env_name, level, version = str(args.env), str(args.level), str(args.version)
    exp_name_suffix = str(args.exp_name_suffix)
    model_type = str(args.model_type)
    group_name = f"{exp_prefix}-{env_name}-{level}-{version}{exp_name_suffix}"
    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"

    ds_dict = get_dataset_dict(env_name)
    assert env_name in ds_dict, ValueError(f"ValueError: env_name = {env_name}")
    gymnasium_env = ds_dict[env_name]["gymnasium_env"]
    render_mode = None if args.render_mode == "" else str(args.render_mode)

    if env_name == "halfcheetah":
        env = gym.make(gymnasium_env, render_mode=render_mode)
        max_ep_len = 1000
        env_targets = [12000.0, 6000.0]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
    elif env_name == "hopper":
        env = gym.make(gymnasium_env, render_mode=render_mode)
        max_ep_len = 1000
        env_targets = [3600.0, 1800.0]
        scale = 1000.0
    elif env_name == "walker2d":
        env = gym.make(gymnasium_env, render_mode=render_mode)
        max_ep_len = 1000
        env_targets = [5000.0, 2500.0]
        scale = 1000.0
    # elif env_name == "ant":
    #     env = gym.make(gymnasium_env, render_mode=render_mode)
    #     max_ep_len = 1000
    #     env_targets = [4200.0, 2100.0]
    #     scale = 1000.0
    # elif env_name == "reacher2d":
    #     from decision_transformer.envs.reacher_2d import Reacher2dEnv
    #
    #     env = Reacher2dEnv()
    #     max_ep_len = 100
    #     env_targets = [76.0, 40.0]
    #     scale = 10.0
    else:
        raise NotImplementedError

    """
    Reward statistics of the offline datasets used in Decision Transformer (provided by d4rl):
        HalfCheetah-v2:
            random: min = -525.9597, max = -85.358215 (mean = -288.79712, std = 80.43068)
            medium: min = -310.2342, max = 5309.3794 (mean = 4770.335, std = 355.7504)
            expert: min = 2045.8278, max = 11252.035 (mean = 10656.426, std = 441.6827)
        Hopper-v2:
            random: min = 2.9301534, max = 292.5542 (mean = 18.398905, std = 17.45116)
            medium: min = 315.868, max = 3222.3606 (mean = 1422.0562, std = 378.9537)
            expert: min = 1645.2765, max = 3759.0837 (mean = 3511.3577, std = 328.58597)
        Walker2D-v2:
            random: min = -17.005825, max = 75.03457 (mean = 1.871351, std = 5.8127255)
            medium: min = -6.605672, max = 4226.94 (mean = 2852.0884, std = 1095.4434)
            expert: min = 763.41614, max = 5011.6934 (mean = 4920.507, std = 136.39494)
    """

    assert scale > 0.0 and int(scale) > 0, ValueError(f"ValueError: scale = {scale}")
    if model_type == "bc":
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.unwrapped.observation_space.shape[0]
    action_dim = env.unwrapped.action_space.shape[0]

    # Load the offline dataset (trajectories)
    dataset_fp = os.path.join(args.save_dir, "parsed", f"{env_name}-{level}-v2.hdf5.pkl")
    assert os.path.isfile(dataset_fp), f"Path error: can not find {dataset_fp}"
    with open(dataset_fp, "rb") as fp_in:
        trajectories = pickle.load(fp_in)

    # Save all trajectory information into separate lists
    mode = str(args.mode)
    states, traj_lens, rewards = [], [], []
    for traj in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            traj["rewards"][-1] = traj["rewards"].sum()
            traj["rewards"][:-1] = 0.
        states.append(traj["observations"])
        traj_lens.append(len(traj["observations"]))
        rewards.append(traj["rewards"].sum())
    traj_lens, rewards = np.array(traj_lens), np.array(rewards)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name} {level}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average rewards: {np.mean(rewards):.2f}, std: {np.std(rewards):.2f}")
    print(f"Max rewards: {np.max(rewards):.2f}, min: {np.min(rewards):.2f}")
    print("=" * 50)

    K = max(int(args.K), 1)
    batch_size = int(args.batch_size)
    num_eval_episodes = int(args.num_eval_episodes)
    pct_traj = float(args.pct_traj)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(rewards)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to re-weight sampling, so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(bsz: int = 256, max_len: int = K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=bsz,
            replace=True,
            p=p_sample,  # re-weights so we sample according to timesteps
        )

        s, a, r, d, rtg, _timesteps, _mask = [], [], [], [], [], [], []  # rtg: returns-to-go / rewards-to-go
        for i in range(bsz):
            _traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, _traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(_traj["observations"][si: si + max_len].reshape(1, -1, state_dim))
            a.append(_traj["actions"][si: si + max_len].reshape(1, -1, action_dim))
            r.append(_traj["rewards"][si: si + max_len].reshape(1, -1, 1))
            if "terminals" in _traj:
                d.append(_traj["terminals"][si: si + max_len].reshape(1, -1))
            else:
                d.append(_traj["dones"][si: si + max_len].reshape(1, -1))
            _timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            _timesteps[-1][_timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(_traj["rewards"][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            traj_len = s[-1].shape[1]  # the length of the trajectory
            s[-1] = np.concatenate([np.zeros((1, max_len - traj_len, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - traj_len, action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - traj_len, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - traj_len)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - traj_len, 1)), rtg[-1]], axis=1) / scale
            _timesteps[-1] = np.concatenate([np.zeros((1, max_len - traj_len)), _timesteps[-1]], axis=1)
            _mask.append(np.concatenate([np.zeros((1, max_len - traj_len)), np.ones((1, traj_len))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        _timesteps = torch.from_numpy(np.concatenate(_timesteps, axis=0)).to(dtype=torch.long, device=device)
        _mask = torch.from_numpy(np.concatenate(_mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, _timesteps, _mask

    def eval_episodes(target_reward: float):

        def eval_func(_model):
            _rewards, _lengths = [], []
            for episode_idx in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == "dt":
                        ret, length = evaluate_episode_dt(
                            env=env,
                            state_dim=state_dim,
                            action_dim=action_dim,
                            model=_model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_reward=target_reward / scale,
                            mode=mode,
                            render_mode=render_mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            seed=seed + episode_idx,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env=env,
                            state_dim=state_dim,
                            action_dim=action_dim,
                            model=_model,
                            max_ep_len=max_ep_len,
                            # scale=1.0,
                            target_reward=target_reward / scale,
                            # mode=mode,
                            render_mode=render_mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            seed=seed + episode_idx,
                            device=device,
                        )
                _rewards.append(ret)
                _lengths.append(length)
            return {
                f"target_{target_reward}_reward_mean": np.mean(_rewards),
                f"target_{target_reward}_reward_std": np.std(_rewards),
                f"target_{target_reward}_length_mean": np.mean(_lengths),
                f"target_{target_reward}_length_std": np.std(_lengths),
            }

        return eval_func

    if model_type == "dt":
        model = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=int(args.embed_dim),
            n_layer=int(args.n_layer),
            n_head=int(args.n_head),
            n_inner=4 * int(args.embed_dim),
            activation_function=args.activation_function,
            n_positions=1024,
            resid_pdrop=float(args.dropout),
            attn_pdrop=float(args.dropout),
        )
    elif model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            action_dim=action_dim,
            max_length=K,
            hidden_size=int(args.embed_dim),
            n_layer=int(args.n_layer),
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = int(args.warmup_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == "dt":
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(env_tgt) for env_tgt in env_targets],
        )
    elif model_type == "bc":
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(env_tgt) for env_tgt in env_targets],
        )
    else:
        raise ValueError(f"ValueError: model_type = {model_type}")

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="decision-transformer",
            config=vars(args)
        )
        # wandb.watch(model)  # wandb has some bug

    for _iter in range(int(args.max_iters)):
        output_logs = trainer.train_iteration(
            num_steps=int(args.num_steps_per_iter),
            iter_num=_iter + 1,
            verbose=bool(args.verbose)
        )
        if log_to_wandb:
            wandb.log(output_logs)


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
    parser.add_argument("--env", type=str, default="hopper",
                        help="Gym env name: \"halfcheetah\", \"hopper\", \"walker2d\"")
    parser.add_argument("--level", type=str, default="random",
                        help="Dataset level: \"random\", \"medium\", \"expert\", \"medium-replay\", \"medium-expert\"")
    parser.add_argument("--version", type=str, default="v2",
                        help="Offline data version: \"v2\", \"v1\", \"v0\" (and \"v4\")")
    # parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device(s), e.g., 0 OR 0,1")
    # parser.add_argument("--log_to_wandb", "-w", type=bool, default=False)
    parser.add_argument("--log_to_wandb", action="store_true", default=False)
    parser.add_argument("--exp_name_suffix", type=str, default="",
                        help="The suffix of the experiment name (for wandb)")

    # Decision Transformer
    parser.add_argument("--render_mode", type=str, default="",
                        help="Gym render_mode: \"\" for None or \"human\" for Human visualization")  # Gym render_mode
    parser.add_argument("--mode", type=str, default="normal")  # normal for standard setting, delayed for sparse
    parser.add_argument("--model_type", type=str, default="dt")  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()
    logger.info(args)
    args.logger = logger

    timer_start = time.perf_counter()

    # Set the random seed of all modules
    args.seed = int(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    # args.env_list = ["halfcheetah", "hopper", "walker2d", "ant"]
    # args.levels = ["random", "medium", "expert", "medium-replay", "medium-expert"]
    args.env_list = ["halfcheetah", "hopper", "walker2d"]
    args.levels = ["random", "medium", "expert"]

    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.has_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.has_cuda else "cpu")
    args.gpus = args.cuda.split(",") if "," in args.cuda else [args.cuda]
    args.gpus = [int(gpu_id) for gpu_id in args.gpus]
    args.device_count = int(torch.cuda.device_count())
    args.ddp_able = args.has_cuda and len(args.gpus) > 1 and args.device_count > 1
    if args.verbose:
        logger.info(
            f"HAS_CUDA: {args.has_cuda}; DEVICE: {args.device}; GPUS: {args.gpus}; DDP able: {args.ddp_able}")
        logger.info(f"torch.__version__: {torch.__version__}")
        logger.info(f"torch.version.cuda: {torch.version.cuda}")
        logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        logger.info(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
        logger.info(f"torch.cuda.get_arch_list(): {torch.cuda.get_arch_list()}")
        if args.has_cuda:
            logger.info(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
            logger.info(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

    # run(exp_prefix=f"gym-mujoco-{args.env}-{args.level}-{args.version}")
    run()

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
