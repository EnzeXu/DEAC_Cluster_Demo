#!/bin/bash -l

#SBATCH --job-name="test"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=10GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu



#SBATCH --output="jobs_oe/test-%jo"
#SBATCH --error="jobs_oe/test-%je"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python demo.py --log_path logs/log_test.txt
