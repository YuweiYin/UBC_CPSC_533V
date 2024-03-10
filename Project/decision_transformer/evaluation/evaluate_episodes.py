import numpy as np
import torch


def evaluate_episode(
        env,
        state_dim,
        action_dim,
        model,
        max_ep_len=1000,
        # scale=1.0,
        target_reward=None,
        # mode="normal",
        state_mean=0.,
        state_std=1.,
        device="cuda",
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # state = env.reset()  # obsolete Gym env
    state, _ = env.reset()  # Tuple[observation (ObsType), info (dictionary)]

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_reward = torch.tensor(target_reward, device=device, dtype=torch.float32)
    # sim_states = []

    episode_reward, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, action_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_reward=target_reward,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_reward += reward
        episode_length += 1

        if done:
            break

    return episode_reward, episode_length


def evaluate_episode_dt(
        env,
        state_dim,
        action_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        target_reward=None,
        state_mean=0.,
        state_std=1.,
        mode="normal",
        device="cuda",
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # state = env.reset()  # obsolete Gym env
    state, _ = env.reset()  # Tuple[observation (ObsType), info (dictionary)]
    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_reward = target_reward
    target_reward = torch.tensor(ep_reward, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    # sim_states = []

    episode_reward, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, action_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_reward.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != "delayed":
            pred_reward = target_reward[0, -1] - (reward / scale)
        else:
            pred_reward = target_reward[0, -1]
        target_reward = torch.cat(
            [target_reward, pred_reward.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_reward += reward
        episode_length += 1

        if done:
            break

    return episode_reward, episode_length
