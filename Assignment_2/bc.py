import os
import time

import numpy as np
import torch
from torch.utils import data

import gymnasium as gym

from eval_policy import eval_policy, device
from model import MyModel
from dataset import Dataset

# import ipdb

# ipdb.set_trace()

BATCH_SIZE = 64
TOTAL_EPOCHS = 100
LEARNING_RATE = 10e-4
PRINT_INTERVAL = 500
TEST_INTERVAL = 2
ENV_NAME = "CartPole-v1"

dataset = Dataset(data_path="./{}_dataset.pkl".format(ENV_NAME))
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

env = gym.make(ENV_NAME)

# TODO INITIALIZE YOUR MODEL HERE
# model = None
model = MyModel(state_size=env.observation_space.shape[0], action_size=env.action_space.n).to(device)


def train_behavioral_cloning():
    timer_start = time.perf_counter()

    # TODO CHOOSE A OPTIMIZER AND A LOSS FUNCTION FOR TRAINING YOUR NETWORK
    # optimizer = None
    # loss_function = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()

    gradient_steps = 0
    score_list = []
    score_best = 0.0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        for iteration, _data in enumerate(dataloader):
            _data = {k: v.to(device) for k, v in _data.items()}

            output = model(_data["state"])

            loss = loss_function(output, _data["action"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gradient_steps % PRINT_INTERVAL == 0:
                print("[epoch {:4d}/{}] [iter {:7d}] [loss {:.5f}]"
                      .format(epoch, TOTAL_EPOCHS, gradient_steps, loss.item()))

            gradient_steps += 1

        if epoch % TEST_INTERVAL == 0:
            score = eval_policy(policy=model, env=ENV_NAME)
            score_list.append(score)
            print("[Test on environment] [epoch {}/{}] [score {:.2f}]"
                  .format(epoch, TOTAL_EPOCHS, score))
            if score > score_best:
                score_best = score
                model_name = "model_best-behavioral_cloning-{}.pt".format(ENV_NAME)
                print("Saving the best model (score: {}) at epoch {} as {}".format(score, epoch, model_name))
                torch.save(model.state_dict(), os.path.join("ckpt", model_name))

    score_max, score_min = np.max(score_list), np.min(score_list)
    score_avg, score_std = np.mean(score_list), np.std(score_list)
    print(f"[END] Score Statistics: Avg = {score_avg}; Max = {score_max}; Min = {score_min}; Std = {score_std}")

    model_name = "model_last-behavioral_cloning-{}.pt".format(ENV_NAME)
    print("Saving the last model as {}".format(model_name))
    torch.save(model.state_dict(), os.path.join("ckpt", model_name))

    timer_end = time.perf_counter()
    time_sec, time_min = timer_end - timer_start, (timer_end - timer_start) / 60
    print(f"\nDONE - Running Time: {time_sec:.1f} sec ({time_min:.1f} min)")


if __name__ == "__main__":
    train_behavioral_cloning()
