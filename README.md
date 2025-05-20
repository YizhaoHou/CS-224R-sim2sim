# CS-224R-sim2sim





## Isaac Gym Installation (WSL2)

> This guide assumes you're using **WSL2 with Ubuntu 20.04+**, **NVIDIA GPU**, and **CUDA 12.1+**. Only **headless (no GUI)** mode is supported in WSL2.

### 1. Prerequisites

- **WSL2 enabled** with Ubuntu installed.
- **NVIDIA driver with WSL2 support** installed on Windows:  
  [WSL2 + CUDA installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- **Miniconda or Anaconda installed** inside WSL2

### 2. Download Isaac Gym

- Go to https://developer.nvidia.com/isaac-gym and download IsaacGym Preview 4.
- Move the .tar.gz file into your WSL2 home directory and extract it:
  ```bash
  tar -xzvf IsaacGym_Preview_4_Package.tar.gz
  ```

### 3. Set up Conda environment (Python 3.8)
```bash
conda create -n isaacgym python=3.8
conda activate isaacgym
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
cd isaacgym/python
pip install -e .
```

### 4. Run Examples:
```bash
cd examples
python 1080_balls_of_solitude.py
```



 
