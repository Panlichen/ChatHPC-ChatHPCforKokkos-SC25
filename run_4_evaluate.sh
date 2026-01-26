#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_h200

bash 1_setup.sh
bash 4_evaluate.sh