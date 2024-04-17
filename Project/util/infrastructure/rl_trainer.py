from collections import OrderedDict
import pickle
import os
import sys
import pdb
import time
import random
import logging

import h5py
import numpy as np
import torch
# import gym
# from gym import wrappers
import gymnasium as gym
from gymnasium import wrappers

from ..infrastructure import pytorch_util as ptu
from ..infrastructure.atari_wrappers import ReturnWrapper

from ..infrastructure.utils import Path
from ..infrastructure import utils
from ..infrastructure.logger import Logger

from ..agents.dqn_agent import DQNAgent
from ..agents.sac_agent import SACAgent
from ..infrastructure.dqn_utils import get_wrapper_by_name, register_custom_envs

# how many rollouts to save as videos to tensorboard
MAX_N_VIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RLTrainer(object):

    def __init__(self, params):
        # Initialization

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params["logdir"])

        # Set random seeds
        seed = self.params["seed"]
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # ptu.init_gpu(
        #     use_gpu=not self.params["no_gpu"],
        #     gpu_id=self.params["which_gpu"]
        # )
        ptu.init_gpu(
            dev=self.params["device"],
        )

        self.env_name = str(self.params["env_name"])
        # "HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"
        # If the model achieves medium_level/expert_level average eval rewards,
        #     then save it as medium-level/expert-level policy
        #     and use it to generate medium-level/expert-level offline datasets/trajectories
        match self.env_name:
            case "HalfCheetah-v4":
                self.random_level = [50.0, 200.0]
                self.medium_level = [3000.0, 7000.0]
                self.expert_level = [8000.0, 12000.0]
            case "Hopper-v4":
                self.random_level = [50.0, 200.0]
                self.medium_level = [1400.0, 1800.0]
                self.expert_level = [2800.0, 3200]
            case "Walker2d-v4":
                self.random_level = [50.0, 200.0]
                self.medium_level = [1700.0, 2300.0]
                self.expert_level = [3500.0, 4500.0]
            case _:
                raise ValueError(f"ValueError: env_name == {self.env_name}")

        # Make the gym environment
        register_custom_envs()
        if self.params["agent_class"] is SACAgent:
            self.env = gym.make(self.env_name, max_episode_steps=self.params["ep_len"])
        else:
            self.env = gym.make(self.env_name)
        if self.params["video_log_freq"] > 0:
            self.episode_trigger = lambda episode: episode % self.params["video_log_freq"] == 0
        else:
            self.episode_trigger = lambda episode: False
        if "env_wrappers" in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.RecordEpisodeStatistics(self.env, deque_size=1000)
            self.env = ReturnWrapper(self.env)
            self.env = wrappers.RecordVideo(
                self.env, os.path.join(self.params["logdir"], "gym"), episode_trigger=self.episode_trigger)
            self.env = params["env_wrappers"](self.env)
            self.mean_episode_reward = -float("nan")
            self.best_mean_episode_reward = -float("inf")
        if "non_atari_colab_env" in self.params and self.params["video_log_freq"] > 0:
            self.env = wrappers.RecordVideo(
                self.env, os.path.join(self.params["logdir"], "gym"), episode_trigger=self.episode_trigger)
            self.mean_episode_reward = -float("nan")
            self.best_mean_episode_reward = -float("inf")

        # self.env.seed(seed)
        self.env.reset(seed=seed)

        # import plotting (locally if "obstacles" env)
        if self.env_name != "obstacles-cs285-v0":
            import matplotlib
            matplotlib.use("Agg")

        # Maximum length for episodes
        self.params["ep_len"] = self.params["ep_len"] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params["ep_len"]

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params["agent_params"]["discrete"] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim

        # simulation timestep, will be used for video saving
        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif "env_wrappers" in self.params:
            self.fps = 30  # This is not actually used when using the Monitor wrapper
        elif "video.frames_per_second" in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata["video.frames_per_second"]
        else:
            self.fps = 10

        # Agent
        agent_class = self.params["agent_class"]
        self.agent = agent_class(self.env, self.params["agent_params"])
        assert isinstance(self.agent.actor, torch.nn.Module) and isinstance(self.agent.critic, torch.nn.Module)

        if self.params["load_ckpt"]:
            # Load the pretrained actor and critic models if they exist
            self.actor_ckpt = self.params["actor_ckpt"]  # The checkpoint path of Actor model
            if isinstance(self.actor_ckpt, str) and len(self.actor_ckpt) > 0 and os.path.isfile(self.actor_ckpt):
                actor_state = torch.load(self.actor_ckpt)
                self.agent.actor.load_state_dict(actor_state)
            self.critic_ckpt = self.params["critic_ckpt"]  # The checkpoint path of Critic model
            if isinstance(self.critic_ckpt, str) and len(self.critic_ckpt) > 0 and os.path.isfile(self.critic_ckpt):
                critic_state = torch.load(self.critic_ckpt)
                self.agent.critic.load_state_dict(critic_state)

        # The directory to save checkpoints
        self.ckpt_dir = os.path.join(self.params["logdir"], self.params["ckpt_dir"])
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Other parameters
        self.log_video = False
        self.log_metrics = False
        self.total_env_steps = 0
        self.start_time = time.perf_counter()
        self.initial_return = 0.0

    def run_training_loop(
            self,
            n_iter,
            collect_policy,
            eval_policy,
            initial_expert_data=None,
            relabel_with_expert: bool = False,
            start_relabel_with_expert: int = 1,
            expert_policy=None,
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expert_data:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_env_steps = 0
        self.start_time = time.perf_counter()

        print_period = 1000 if isinstance(self.agent, DQNAgent) else max(1, self.params["scalar_log_freq"])

        for itr in range(n_iter):
            if itr % print_period == 0:
                print(f"\n********** Iteration {itr} ************")

            # Decide if videos should be rendered/logged at this iteration
            if self.params["video_log_freq"] != -1 and itr % self.params["video_log_freq"] == 0:
                self.log_video = True
            else:
                self.log_video = False

            # Decide if metrics should be logged
            if self.params["scalar_log_freq"] == -1:
                self.log_metrics = False
            elif itr % self.params["scalar_log_freq"] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent):
                # only perform an env step and add to replay buffer for DQN
                self.agent.step_env()
                env_steps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                use_batch_size = self.params["batch_size"]
                if itr == 0:
                    use_batch_size = self.params["batch_size_initial"]
                paths, env_steps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expert_data, collect_policy, use_batch_size)
                )

            self.total_env_steps += env_steps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer (add rollouts)
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            # if itr % print_period == 0:
            #     print("Training agent...")
            all_logs = self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                # print("Logging...")
                if isinstance(self.agent, DQNAgent):
                    self.perform_dqn_logging(all_logs)
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

    def run_sac_training_loop(
            self,
            n_iter,
            collect_policy,
            eval_policy
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        """

        # init vars at beginning of training
        self.total_env_steps = 0
        self.start_time = time.perf_counter()
        episode_step = 0
        episode_return = 0
        episode_stats = {"reward": [], "ep_len": []}

        done = False
        # print_period = 1000
        print_period = 1000 if isinstance(self.agent, DQNAgent) else max(1, self.params["scalar_log_freq"])

        for itr in range(n_iter):
            if itr % print_period == 0:
                print(f"\n********** Iteration {itr} ************")

            # Decide if videos should be rendered/logged at this iteration
            if self.params["video_log_freq"] != -1 and itr % self.params["video_log_freq"] == 0:
                self.log_video = True
            else:
                self.log_video = False

            # Decide if metrics should be logged
            if self.params["scalar_log_freq"] == -1:
                self.log_metrics = False
            elif itr % self.params["scalar_log_freq"] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            use_batch_size = self.params["batch_size"]
            train_video_paths = None
            if itr == 0:
                use_batch_size = self.params["batch_size_initial"]
                # print("Sampling seed steps for training...")
                paths, env_steps_this_batch = utils.sample_random_trajectories(
                    self.env, use_batch_size, self.params["ep_len"])
                train_video_paths = None
                episode_stats["reward"].append(np.mean([np.sum(path["reward"]) for path in paths]))
                episode_stats["ep_len"].append(len(paths[0]["reward"]))
                self.total_env_steps += env_steps_this_batch
            else:
                if itr == 1 or done:
                    obs, _ = self.env.reset()
                    episode_stats["reward"].append(episode_return)
                    episode_stats["ep_len"].append(episode_step)
                    episode_step = 0
                    episode_return = 0

                action = self.agent.actor.get_action(obs)[0]
                # next_obs, reward, done, _ = self.env.step(action)
                next_obs, reward, done, truncated, info = self.env.step(action)

                episode_return += reward

                episode_step += 1
                self.total_env_steps += 1

                if done:
                    terminal = 1
                else:
                    terminal = 0
                paths = [Path([obs], [], [action], [reward], [next_obs], [terminal])]
                obs = next_obs

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            # if itr % print_period == 0:
            #     print("Training agent...")
            all_logs = self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                print("Logging...")
                self.perform_sac_logging(itr, episode_stats, eval_policy, train_video_paths, all_logs)
                episode_stats = {"reward": [], "ep_len": []}

    def collect_training_trajectories(
            self,
            itr,
            initial_expert_data,
            collect_policy,
            num_transitions_to_sample,
            save_expert_data_to_disk: bool = False,
    ):
        """
        :param itr:
        :param initial_expert_data:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :param save_expert_data_to_disk:  [bool]
        :return:
            paths: a list trajectories
            env_steps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # Decide whether to load training data or use the current policy to collect more data
        # HINT: depending on if it"s the first iteration or not, decide whether to either
        # (1) load the data. In this case you can directly return as follows
        # ``` return loaded_paths, 0, None ```
        if itr == 0 and initial_expert_data:
            with open(initial_expert_data, "rb") as f:
                loaded_paths = pickle.load(f)
                return loaded_paths, 0, None
                # (2) collect `self.params["batch_size"]` transitions

        # Collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params["ep_len"]

        # print("Collecting data for training...")
        paths, env_steps_this_batch = utils.sample_trajectories(
            self.env, collect_policy, num_transitions_to_sample, self.params["ep_len"])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_N_VIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            # print("Collecting train rollouts to be used for saving videos...")
            # Look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_N_VIDEO, MAX_VIDEO_LEN, True)

        return paths, env_steps_this_batch, train_video_paths

    def train_agent(self):
        # Get this from hw1 or hw2
        # print("Training agent using sampled data from replay buffer...")
        all_logs = []
        for train_step in range(self.params["num_agent_train_steps_per_iter"]):
            # Sample some data from the data buffer
            # HINT1: use the agent"s sample function
            # HINT2: how much data = self.params["train_batch_size"]
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params["train_batch_size"])

            # Use the sampled data to train an agent
            # HINT: use the agent"s train function
            # HINT: keep the agent"s training log for debugging
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)

        return all_logs

    def save_agent(self, itr: int, level: str = "random"):
        if isinstance(self.agent, torch.nn.Module):
            self.agent.save("{}/{}_{}---agent_itr_{}.pt".format(
                self.ckpt_dir, self.env_name, level, itr))
        if hasattr(self.agent, "actor") and isinstance(self.agent.actor, torch.nn.Module):
            self.agent.actor.save("{}/{}_{}---agent_actor_itr_{}.pt".format(
                self.ckpt_dir, self.env_name, level, itr))
        if hasattr(self.agent, "critic") and isinstance(self.agent.critic, torch.nn.Module):
            self.agent.critic.save("{}/{}_{}---agent_critic_itr_{}.pt".format(
                self.ckpt_dir, self.env_name, level, itr))

    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = self.env.get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(float(self.best_mean_episode_reward), float(self.mean_episode_reward))

        logs = OrderedDict()

        logs["Train_EnvStepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.perf_counter() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print("{} : {}".format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)

        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):
        last_log = all_logs[-1]

        # collect eval trajectories, for logging
        # print("Collecting data for eval...")
        eval_paths, eval_env_steps_this_batch = utils.sample_trajectories(
            self.env, eval_policy, self.params["eval_batch_size"], self.params["ep_len"])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths is not None:
            # print("Collecting video rollouts eval")
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_N_VIDEO, MAX_VIDEO_LEN, True)

            # save train/eval videos
            print("Saving train rollouts as videos...")
            self.logger.log_paths_as_videos(
                train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_N_VIDEO, video_title="train_rollouts")
            self.logger.log_paths_as_videos(
                eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_N_VIDEO, video_title="eval_rollouts")

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            if len(eval_returns) > 0:
                logs["Eval_AverageReturn"] = np.mean(eval_returns)
                logs["Eval_StdReturn"] = np.std(eval_returns)
                logs["Eval_MaxReturn"] = np.max(eval_returns)
                logs["Eval_MinReturn"] = np.min(eval_returns)
            if len(eval_ep_lens) > 0:
                logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            if len(train_returns) > 0:
                logs["Train_AverageReturn"] = np.mean(train_returns)
                logs["Train_StdReturn"] = np.std(train_returns)
                logs["Train_MaxReturn"] = np.max(train_returns)
                logs["Train_MinReturn"] = np.min(train_returns)
            if len(train_ep_lens) > 0:
                logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvStepsSoFar"] = self.total_env_steps
            logs["TimeSinceStart"] = time.perf_counter() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)

            self.logger.flush()

            # Save models
            avg_eval_rewards = float(np.mean(eval_returns))
            if self.params["save_params"]:
                if self.random_level[0] <= avg_eval_rewards <= self.random_level[1]:
                    logging.info(f"Save the random-level agent: itr = {itr}; avg_eval_rewards = {avg_eval_rewards}")
                    self.save_agent(itr=itr, level="random")
                if self.medium_level[0] <= avg_eval_rewards <= self.medium_level[1]:
                    logging.info(f"Save the medium-level agent: itr = {itr}; avg_eval_rewards = {avg_eval_rewards}")
                    self.save_agent(itr=itr, level="medium")
                if self.expert_level[0] <= avg_eval_rewards <= self.expert_level[1]:
                    logging.info(f"Save the expert-level agent: itr = {itr}; avg_eval_rewards = {avg_eval_rewards}")
                    self.save_agent(itr=itr, level="expert")

    def perform_sac_logging(self, itr, stats, eval_policy, train_video_paths, all_logs):
        last_log = all_logs[-1]

        # collect eval trajectories, for logging
        # print("Collecting data for eval...")
        eval_paths, eval_env_steps_this_batch = utils.eval_trajectories(
            self.env, eval_policy, self.params["eval_batch_size"], self.params["ep_len"])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths is not None:
            # print("Collecting video rollouts eval")
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_N_VIDEO, MAX_VIDEO_LEN, True)

            # save train/eval videos
            print("Saving train rollouts as videos...")
            self.logger.log_paths_as_videos(
                train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_N_VIDEO, video_title="train_rollouts")
            self.logger.log_paths_as_videos(
                eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_N_VIDEO, video_title="eval_rollouts")

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            if len(eval_returns) > 0:
                logs["Eval_AverageReturn"] = np.mean(eval_returns)
                logs["Eval_StdReturn"] = np.std(eval_returns)
                logs["Eval_MaxReturn"] = np.max(eval_returns)
                logs["Eval_MinReturn"] = np.min(eval_returns)
            if len(eval_ep_lens) > 0:
                logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            if len(stats["reward"]) > 0:
                logs["Train_AverageReturn"] = np.mean(stats["reward"])
                logs["Train_StdReturn"] = np.std(stats["reward"])
                logs["Train_MaxReturn"] = np.max(stats["reward"])
                logs["Train_MinReturn"] = np.min(stats["reward"])

            if len(stats["ep_len"]) > 0:
                logs["Train_AverageEpLen"] = np.mean(stats["ep_len"])

            logs["Train_EnvStepsSoFar"] = self.total_env_steps
            logs["TimeSinceStart"] = time.perf_counter() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(stats["reward"])
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                try:
                    self.logger.log_scalar(value, key, itr)
                except Exception as e:
                    print(e)
                    pdb.set_trace()

            self.logger.flush()

            # Save models
            avg_eval_rewards = float(np.mean(eval_returns))
            if self.params["save_params"]:
                if self.random_level[0] <= avg_eval_rewards <= self.random_level[1]:
                    logging.info(f"Save the random-level agent: itr = {itr}; avg_eval_rewards = {avg_eval_rewards}")
                    self.save_agent(itr=itr, level="random")
                if self.medium_level[0] <= avg_eval_rewards <= self.medium_level[1]:
                    logging.info(f"Save the medium-level agent: itr = {itr}; avg_eval_rewards = {avg_eval_rewards}")
                    self.save_agent(itr=itr, level="medium")
                if self.expert_level[0] <= avg_eval_rewards <= self.expert_level[1]:
                    logging.info(f"Save the expert-level agent: itr = {itr}; avg_eval_rewards = {avg_eval_rewards}")
                    self.save_agent(itr=itr, level="expert")

    def collect_offline_dataset(
            self,
            save_dir: str,
            env_name: str = "HalfCheetah",  # "HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "Ant-v4"
            env_version: str = "v4",
            level: str = "random",
            max_n_traj: int = 1000000,
            max_traj_len: int = 150,
    ):
        actions = []
        observations = []
        next_observations = []
        rewards = []
        terminals = []
        timeouts = []

        obs, _ = self.env.reset()
        for _ in range(max_n_traj):
            for _ in range(max_traj_len):
                # self.env.render()
                action = self.agent.actor.get_action(obs)[0]
                next_obs, reward, done, truncated, info = self.env.step(action)
                actions.append(action)
                observations.append(obs)
                next_observations.append(next_obs)
                rewards.append(reward)
                terminals.append(done)
                timeouts.append(False)

                if done:
                    break

        save_fp = os.path.join(save_dir, f"{env_name}-{env_version}_{level}.hdf5")
        with h5py.File(save_fp, "w") as fp_out:
            fp_out.create_dataset("actions", data=np.array(actions), dtype="f4")
            fp_out.create_dataset("observations", data=np.array(observations), dtype="f4")
            fp_out.create_dataset("next_observations", data=np.array(next_observations), dtype="f4")
            fp_out.create_dataset("rewards", data=np.array(rewards), dtype="f4")
            fp_out.create_dataset("terminals", data=np.array(terminals), dtype="bool")
            fp_out.create_dataset("timeouts", data=np.array(timeouts), dtype="bool")
