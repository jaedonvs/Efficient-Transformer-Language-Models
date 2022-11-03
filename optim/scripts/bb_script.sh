#!/bin/bash
#PBS -P CSCI1335
#PBS -N BB_Transformer_Sweep
#PBS -q gpu_1
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -l walltime=12:00:00
#PBS -m abe
#PBS -WMail_Users=VSCJAE001@myuct.ac.za
#PBS
ulimit -s unlimited

cd ${HOME}
module purge
module load chpc/python/anaconda/3-2021.05
module load chpc/cuda/11.2/PCIe/11.2
module load gcc/9.2.0
cd lustre/optim || exit

now=$(date +"%Y%m%d_%H%M%S")
start=`date +%s`

#run python script
echo "Running python script"
python -u trainBB.py 

end=`date +%s`
runtime=$(expr $end - $start)
echo "Runtime: ${runtime} seconds" &>> ${out_dir}/log
