#!/bin/bash

#$ -cwd -V
#$ -S /bin/bash
#$ -N run_model.sh
#$ -q short.qc
#$ -o logs/
#$ -e logs/

# activate the appropriate python environment ...
# ... and submit job as: qsub -t 1-729 run_model.sh
#
# to concatenate all metrics run: awk 'FNR>1 || NR==1' results/simdata{1..729}.csv > results/metrics.csv

#export OPENBLAS_NUM_THREADS=1
#export OMP_NUM_THREADS=1

echo '*************************************************'
echo 'SGE job ID: '$JOB_ID
echo 'SGE task ID: '$SGE_TASK_ID
echo 'Run on host: '`hostname`
echo 'Operating system: '`uname -s`
echo 'Username: '`whoami`
echo 'Started at: '`date`
echo 'OPENBLAS_NUM_THREADS: '$OPENBLAS_NUM_THREADS
echo 'OMP_NUM_THREADS: '$OMP_NUM_THREADS
echo '*************************************************'

python run_models_on_simulated_data.py $SGE_TASK_ID

echo '*************************************************'
echo 'Finished at: '`date`
echo '*************************************************'
