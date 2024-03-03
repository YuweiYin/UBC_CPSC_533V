# UBC CPSC 532V (2023W2) Assignment 3

- Assignment 3: [Task](./hw3_policy_gradients.ipynb) and [GitHub](https://github.com/UBCMOCCA/CPSC533V_2023W2/tree/main/A3)
- [Assignment 3 Jupyter Notebook](./hw3_policy_gradients-JC_YY.ipynb) - [Juntai Cao](https://github.com/juntaic7) and [Yuwei Yin](https://github.com/YuweiYin)

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

Install `swig` and `gymnasium[box2d]`:

```bash
conda install -c miniconda swig
pip install 'gymnasium[box2d]' 
```

## 2. Run the code

### 2.1. Jupyter Notebook

```bash
jupyter notebook
```

Run `hw3_policy_gradients-JC_YY.ipynb`

---
