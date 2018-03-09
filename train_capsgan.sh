#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o sample_experiment_outfile  # send stdout to sample_experiment_outfile
#SBATCH -e sample_experiment_errfile  # send stderr to sample_experiment_errfile
#SBATCH -t 2:00:00  # time requested in hour:minute:secon

export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

# Activate experiment variables -- Change the name experiment for something meaningful
mkdir -p mnist_dir/experiment

TRAIN_DIR=mnist_dir/experiment/train
EVAL_DIR=mnist_dir/experiment/eval

# After training I'd recomment move logs, error/output files and such inside the experiment folder

mkdir -p $TRAIN_DIR
mkdir -p $EVAL_DIR

# To change training size just add the --train_size label, e.g. --train_size=200000

python main.py --dataset mnist --input_height=28 --output_height=28 --checkpoint_dir=$TRAIN_DIR --sample_dir=$EVAL_DIR --train --model capsgan
