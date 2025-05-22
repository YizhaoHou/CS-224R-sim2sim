# Download Miniforge3
wget https://github.com/conda-forge/miniforge/releases/download/25.3.0-1/Miniforge3-25.3.0-1-Linux-x86_64.sh

# Install Miniforge3
yes | bash Miniforge3-25.3.0-1-Linux-x86_64.sh -b -p $HOME/miniforge3

# Add Miniforge3 to PATH
echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create a new conda environment
conda create -n unitree_rl python=3.8 -y
echo "conda activate unitree_rl" >> ~/.bashrc
conda activate unitree_rl

# Install dependencies
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install issac gym 
wget https://developer.nvidia.com/isaac-gym-preview-4
mv isaac-gym-preview-4 isaac-gym-preview-4.tar.gz
tar -xvzf isaac-gym-preview-4.tar.gz
cd isaacgym/python
pip install -e .
cd ../..

# Install rsl_rl
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout v1.0.2
pip install -e .
cd ..

# Install unitree_gym
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
cd unitree_rl_gym
pip install -e .
cd ..


# avoid the error: libpython3.8.so: cannot open shared object file: No such file or directory
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

