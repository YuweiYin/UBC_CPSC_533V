# UBC CPSC 533V (2023W2) Project

* [Project Report](./docs/UBC_CPSC_533V-Project_Report.pdf)

## Environment (Linux; macOS)

### Miniconda3

```bash
# https://docs.conda.io/projects/miniconda/en/latest/
mkdir -p ~/miniconda3
#curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Then, reopen the terminal.

### Python3 Virtual Environments

Now, create the conda venv (do not use the most-updated Python3 version):

```bash
conda create -n 533v -y python=3.10
conda activate 533v
```

### Python3 Packages

```bash
pip install -r requirements.txt
```

```bash
# brew install swig  # MacOSX
pip install box2d-py
```

### MuJoCo Env and Offline Data

* **GYM**: [OpenAI gym](https://github.com/openai/gym), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
* **MuJoCo**: [Google DeepMind](https://github.com/google-deepmind/mujoco/), [OpenAI](https://github.com/openai/mujoco-py)
* **Decision Transformer**: [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html), [Code](https://github.com/kzl/decision-transformer)

We modified the official implementations of [Decision Transformer](https://github.com/kzl/decision-transformer), 
[mujoco-py](https://github.com/openai/mujoco-py), and [D4RL](https://github.com/Farama-Foundation/D4RL)
in our re-implementation.

The original [code](https://github.com/kzl/decision-transformer/blob/master/gym/data/download_d4rl_datasets.py)
to obtain the offline datasets/trajectories requires C compiling in the usage of `mujoco-py` and `d4rl`:

```bash
#pip3 install -U 'mujoco-py<2.2,>=2.1'
#pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

#cd ~
#mkdir .mujoco
#cd .mujoco
#wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
#tar -zxvf mujoco210-linux-x86_64.tar.gz

#brew install gcc
```

The [download process](https://github.com/kzl/decision-transformer/blob/master/gym/data/download_d4rl_datasets.py)
is a bit buggy, so (in [our code](./get_offline_data.py))
we directly download the `DATASET_URLS` in the `data/data_infos.py` to `data/raw/`
and parse the HDF5 files to `data/parsed/`.

Each data point has `"actions", "observations", "next_observations", "rewards", "terminals", "timeouts"` attributes,
and each trajectory (RL data path) will be a chunk of the whole trajectory.


## Run Decision Transformer

### Step 1: Get the offline RL data

We download and parse the offline data of four [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) tasks:

* [**Half Cheetah**](https://gymnasium.farama.org/environments/mujoco/half_cheetah/):
  * Action Space: `Box(-1.0, 1.0, (6,), float32)`
  * Observation Space: `Box(-inf, inf, (17,), float64)`
  * import: `gymnasium.make("HalfCheetah-v4")`
* [**Walker2D**](https://gymnasium.farama.org/environments/mujoco/walker2d/)
  * Action Space: `Box(-1.0, 1.0, (6,), float32)`
  * Observation Space: `Box(-inf, inf, (17,), float64)`
  * import: `gymnasium.make("Walker2d-v4")`
* [**Hopper**](https://gymnasium.farama.org/environments/mujoco/hopper/)
  * Action Space: `Box(-1.0, 1.0, (3,), float32)`
  * Observation Space: `Box(-inf, inf, (11,), float64)`
  * import: `gymnasium.make("Hopper-v4")`

Each task has five different types (levels) of offline data ([URLs file](./data/data_infos.py)):
`"random"`, `"medium"`, `"expert"`, `"medium-replay"`, and `"medium-expert"`.
We use `"random"`, `"medium"`, and `"expert"`.

The offline data is collected using `v2` version env. Our experiments will be taken in the `v4`version env 
(to avoid using buggy code in `mujoco-py` and `d4rl`). Despite the version mismatch,
the `states` and `observations` remain in the same dimension and corresponding meaning.

```bash
#brew install wget  # to use `wget` on macOS
#python3 get_offline_data.py
bash get_offline_data.sh
```

Specify the download or parsing task by passing the `--task` parameter:

```bash
python3 get_offline_data.py --task "download"
python3 get_offline_data.py --task "parse"
```

The raw and parsed offline datasets are shared on 
[Google Drive](https://drive.google.com/drive/folders/1iRs7pCTRuoQb6iS6KskCDynxkHDjAXGi?usp=sharing).

### Step 2: Run Gym-MuJoCo experiments

```bash
python3 train_dt_offline.py
```

Specify the Gym env and offline dataset level:

```bash
python3 train_dt_offline.py --env "hopper" --level "random"
```

To show the training/evaluation logs and save logs to [wandb](https://wandb.ai/):

```bash
python3 train_dt_offline.py --env "hopper" --level "random" --verbose --log_to_wandb
```

## Training RL Policy on MuJoCo V4 Env

### Actor Critic (AC)

```bash
python3 train_ac.py --exp_name "AC_Training" --env_name "HalfCheetah-v4" \
  --seed 42 --cuda 0 --use_gpu --verbose \
  --ep_len 200 --n_iter 200 --num_agent_train_steps_per_iter 100 \
  --num_critic_updates_per_agent_update 1 --num_actor_updates_per_agent_update 1 \
  --num_target_updates 10 --num_grad_steps_per_target_update 10 \
  --batch_size 1000 --eval_batch_size 400 --train_batch_size 1000 \
  --n_layers 5 --size 64 \
  --discount 0.99 --learning_rate 5e-3 \
  --scalar_log_freq 10
```

### Soft Actor Critic (SAC)

```bash
python3 train_sac.py --exp_name "SAC_Training" --env_name "HalfCheetah-v4" \
  --seed 42 --cuda 0 --use_gpu --verbose \
  --ep_len 200 --n_iter 200 --num_agent_train_steps_per_iter 100 \
  --num_critic_updates_per_agent_update 10 --num_actor_updates_per_agent_update 10 \
  --actor_update_frequency 1 --critic_target_update_frequency 1 \
  --batch_size 1000 --eval_batch_size 400 --train_batch_size 256 \
  --n_layers 5 --size 64 \
  --discount 0.99 --learning_rate 5e-3 --init_temperature 1.0 \
  --scalar_log_freq 10
```

* Training [log](./log/train_sac.log) (Averaged Evaluation Reward: `Eval_AverageReturn`)

## Collect V4 Offline Datasets using the Trained Policy

```bash
python3 collect_offline_data.py --exp_name "SAC_Training" --env_name "HalfCheetah-v4" \
  --seed 42 --cuda 0 --use_gpu --verbose \
  --ep_len 200 --n_iter 200 --num_agent_train_steps_per_iter 100 \
  --num_critic_updates_per_agent_update 10 --num_actor_updates_per_agent_update 10 \
  --actor_update_frequency 1 --critic_target_update_frequency 1 \
  --batch_size 1000 --eval_batch_size 400 --train_batch_size 256 \
  --n_layers 5 --size 64 \
  --discount 0.99 --learning_rate 5e-3 --init_temperature 1.0 \
  --scalar_log_freq 10 \
  --load_ckpt --actor_ckpt "/path/to/actor_ckpt/" --critic_ckpt "/path/to/critic_ckpt/" \
  --level "ALL" --max_n_traj 1000000 --max_traj_len 150 --data_dir "data/v4_datasets/"
```

---
