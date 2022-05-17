#!/bin/bash -l



#!/bin/bash
#SBATCH --job-name=GPUJOB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account="chenGrp"
#SBATCH --partition=gpu
#SBATCH --mem=120gb
#SBATCH --gres=gpu:V100:1


#SBATCH --output="jobs_oe/test-%jo"
#SBATCH --error="jobs_oe/test-%je"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python demo.py --log_path logs/log_test.txt
