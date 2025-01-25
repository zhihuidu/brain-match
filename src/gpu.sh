#!/bin/bash -l
#SBATCH --job-name=mpigpu
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=zd4 # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=9:59:00  # D-HH:MM:SS

module load CUDA/12.0.0
module load gompi/2022b

echo "start time is"
date
start_time=$(date +%s)
d=`date +%F-%H-%M-%S`
output="../data/mpi-gpu-${SLURM_JOB_ID}-$d.csv"

#mpirun -np 4 ./mpi-gpu  ../data/gm.csv ../data/gf.csv ../data/best.csv $output 2
srun --mpi=pmix  ./gpu  ../data/gm.csv ../data/gf.csv ../data/best.csv $output 4

end_time=$(date +%s)
execution_time=$(( end_time - start_time ))
echo "execution time is $execution_time"

