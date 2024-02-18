# UBC CPSC 532V (2023W2) Assignment 2

- [Assignment 2 Jupyter Notebook](./hw2_Tabular_DQN-Yuwei_Yin.ipynb)

## 1. Environment (Linux; macOS)

### 1.1. Miniconda3

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

### 1.2. Python3 Virtual Environments

Now, create the conda venv (do not use the most-updated Python3 version):

```bash
conda create -n 533v -y python=3.10
conda activate 533v
```

### 1.3. Python3 Packages

```bash
pip install -r requirements.txt
```

## 2. Run the code

### 2.1. Jupyter Notebook

```bash
jupyter notebook
```

Run `HW2_Tabular_DQN-Yuwei_Yin.ipynb`

### 2.2. Behavioral Cloning

```bash
python3 bc.py
#python3 bc.py > log/bc.log
```

### 2.3. Deep Q Learning (without Replay Memory)

```bash
python3 dqn.py
#python3 dqn.py > log/dqn.log
```

### 2.4. Deep Q Learning (with Replay Memory)

```bash
python3 dqn_replay.py
#python3 dqn_replay.py > log/dqn_replay.log
```

## 3. Experimental Results

### 3.1. Training Logs

- **Behavioral Cloning**: [bc.log](./log/bc.log)
- **Deep Q Learning (without Replay Memory)**: [dqn.log](./log/dqn.log)
- **Deep Q Learning (with Replay Memory)**: [dqn_replay.log](./log/dqn_replay.log)

### 3.2. Evaluation

- **Behavioral Cloning**:
```bash
python3 eval_policy.py --model-path "ckpt/model_best-behavioral_cloning-CartPole-v1.pt" --env "CartPole-v1"
```
```txt
[Episode    0/10] [reward 190.0]
[Episode    1/10] [reward 232.0]
[Episode    2/10] [reward 247.0]
[Episode    3/10] [reward 254.0]
[Episode    4/10] [reward 313.0]
[Episode    5/10] [reward 212.0]
[Episode    6/10] [reward 280.0]
[Episode    7/10] [reward 197.0]
[Episode    8/10] [reward 320.0]
[Episode    9/10] [reward 202.0]
```

- **Deep Q Learning (without Replay Memory)**:
```bash
python3 eval_policy.py --model-path "ckpt/model_best-dqn-CartPole-v1.pt" --env "CartPole-v1"
```
```txt
[Episode    0/10] [reward 500.0]
[Episode    1/10] [reward 481.0]
[Episode    2/10] [reward 491.0]
[Episode    3/10] [reward 487.0]
[Episode    4/10] [reward 500.0]
[Episode    5/10] [reward 495.0]
[Episode    6/10] [reward 473.0]
[Episode    7/10] [reward 500.0]
[Episode    8/10] [reward 500.0]
[Episode    9/10] [reward 500.0]
```

- **Deep Q Learning (with Replay Memory)**:
```bash
python3 eval_policy.py --model-path "ckpt/model_best-dqn_reply-CartPole-v1.pt" --env "CartPole-v1"
```
```txt
[Episode    0/10] [reward 500.0]
[Episode    1/10] [reward 500.0]
[Episode    2/10] [reward 500.0]
[Episode    3/10] [reward 500.0]
[Episode    4/10] [reward 500.0]
[Episode    5/10] [reward 500.0]
[Episode    6/10] [reward 500.0]
[Episode    7/10] [reward 500.0]
[Episode    8/10] [reward 500.0]
[Episode    9/10] [reward 500.0]
```

---
