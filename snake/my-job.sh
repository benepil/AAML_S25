#!/bin/bash
#SBATCH --job-name=snake_rl
#SBATCH --output=snake_rl_%j.out
#SBATCH --error=snake_rl_%j.err
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00

# Initialize conda (this loads the base environment and the 'conda' command).
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if the conda environment "snake" exists; if not, create it.
if ! conda env list | grep -q "^snake\s"; then
    echo "Creating conda environment 'snake'..."
    conda create -y -n snake python=3.10
else
    echo "Conda environment 'snake' already exists."
fi

# Activate the "snake" environment.
conda activate snake

# Install required Python packages from requirements.txt.
# Adjust the path to requirements.txt if necessary.
#pip install -r requirements.txt

# Change to the directory where your training script is located.
cd snake-game

# Run your training script.
# The --output_dir parameter saves checkpoints and logs to a directory outside snake-game.
python snake.py --output_dir ../snake_rl_output --num_episodes 100
