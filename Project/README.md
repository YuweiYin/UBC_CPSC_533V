# UBC CPSC 533V (2023W2) Project

## Environment (Linux; macOS)

### Miniconda3

```bash
# https://docs.conda.io/projects/miniconda/en/latest/
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
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


## Run the code

### Step 1: Get the offline RL data

We download and parse the offline data of four [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) tasks:

* [Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/):
  * Action Space: `Box(-1.0, 1.0, (6,), float32)`
  * Observation Space: `Box(-inf, inf, (17,), float64)`
  * import: `gymnasium.make("HalfCheetah-v4")`
* [Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/)
  * Action Space: `Box(-1.0, 1.0, (6,), float32)`
  * Observation Space: `Box(-inf, inf, (17,), float64)`
  * import: `gymnasium.make("Walker2d-v4")`
* [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/)
  * Action Space: `Box(-1.0, 1.0, (3,), float32)`
  * Observation Space: `Box(-inf, inf, (11,), float64)`
  * import: `gymnasium.make("Hopper-v4")`
* [Ant](https://gymnasium.farama.org/environments/mujoco/ant/)
  * Action Space: `Box(-1.0, 1.0, (8,), float32)`
  * Observation Space: `Box(-inf, inf, (27,), float64)`
  * import: `gymnasium.make("Ant-v4")`

Each task has five different types (levels) of offline data ([URLs file](./data/data_infos.py)):
`"random"`, `"medium"`, `"expert"`, `"medium-replay"`, and `"medium-expert"`.

The offline data is collected using `v2` version env. Our experiments will be taken in the `v4`version env 
(to avoid using buggy code in `mujoco-py` and `d4rl`). Despite the version mismatch,
the `states` and `observations` remain in the same dimension and corresponding meaning.

```bash
#brew install wget  # to use `wget` on macOS
python3 get_offline_data.py
```

Specify the download or parsing task by passing the `--task` parameter:

```bash
python3 get_offline_data.py --task "download"
python3 get_offline_data.py --task "parse"
```

### Step 2: Run Gym-MuJoCo experiments

```bash
python3 run_experiment.py
```

---
