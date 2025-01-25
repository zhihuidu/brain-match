#!/bin/bash -l
#SBATCH --job-name=t-gpu
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=zd4 # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-23:59:00  # D-HH:MM:SS
##SBATCH --mem-per-cpu=60000M

#module load DMTCP/3.0.0
module load CUDA/12.4.0

#export DMTCP_DL_PLUGIN=0
echo "start time is"
date
start_time=$(date +%s)
d=`date +%F-%H-%M-%S`
output="../data/gpu-t-${SLURM_JOB_ID}-$d.csv"
#export LD_LIBRARY_PATH=/project/zd4/mpich-4.3.0b1/mpich-install/lib:$LD_LIBRARY_PATH

#dmtcp_launch --interval 85000 ./singlegpu  ../data/gm.csv ../data/gf.csv ../data/best.csv $output
./t  ../data/gm.csv ../data/gf.csv ../data/best.csv $output

end_time=$(date +%s)
execution_time=$(( end_time - start_time ))
echo "execution time is $execution_time"

