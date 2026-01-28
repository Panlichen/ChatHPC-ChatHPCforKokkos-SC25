#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_h100

bash 1_setup.sh
bash 3_verify.sh