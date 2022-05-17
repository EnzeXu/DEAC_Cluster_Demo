#!/bin/bash -l

#SBATCH --job-name="test"
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:V100:1


#SBATCH --output="jobs_oe/test-%jo"
#SBATCH --error="jobs_oe/test-%je"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python demo.py --log_path logs/log_test.txt
