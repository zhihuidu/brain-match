#!/bin/bash
#SBATCH --job-name=mpi_graph_align    # Job name
#SBATCH --nodes=8                 # Number of nodes
#SBATCH --ntasks-per-node=1       # MPI processes per node
#SBATCH --cpus-per-task=128         # OpenMP threads per MPI process
#SBATCH --mem=128G                 # Memory per node
#SBATCH --output=align_%j.out     # Standard output log
#SBATCH --error=align_%j.err      # Standard error log

# Print some debugging information
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE"

module load intel/oneapi/mpi/latest
# Set OpenMP thread number
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

d=`date +%F-%H-%M-%S`
output="../data/realign-parallel-${SLURM_JOB_ID}-$d.csv"
echo $output

mpirun -np 8 ./mpi-omp  ../data/gm.csv ../data/gf.csv ../data/best.csv $output

