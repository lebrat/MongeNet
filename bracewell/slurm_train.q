#!/bin/bash

#SBATCH --time=24:00:00 # walltime
#SBATCH --nodes=1 # Number of computer nodes
#SBATCH --ntasks-per-node=4 # number of process per node
#SBATCH --cpus-per-task=1 # number of threads per process
#SBATCH --mem-per-cpu=32G # memory per node
#SBATCH --gres=gpu:1 # number of gpus

# Load libraries to run the code
SRC_DIR=/scratch1/fon022/MongeNet/
cd ${SRC_DIR}
source /apps/miniconda3/4.3.13/etc/profile.d/conda.sh
conda deactivate
source ${SRC_DIR}/bracewell/setup.sh 

# run train code
python train.py user_config=${CONFIG} trainer.output_dir=${OUT}

# example
#sbatch --export=CONFIG=<path>,OUT=<path> /scratch1/fon022/MongeNet/bracewell/slurm_train.q