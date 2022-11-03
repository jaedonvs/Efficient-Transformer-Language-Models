#!/bin/bash
#PBS -P CSCI1335
#PBS -N bb_small
#PBS -q gpu_1
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -l walltime=12:00:00
#PBS -e /home/jvanschalkwyk/lustre/mlm/bb_err
#PBS -m abe
#PBS -WMail_Users=VSCJAE001@myuct.ac.za
#PBS
ulimit -s unlimited

cd ${HOME}
module purge
module load chpc/python/anaconda/3-2021.05
module load chpc/cuda/11.2/PCIe/11.2
module load gcc/9.2.0
cd lustre/mlm || exit

out_dir=${HOME}/lustre/mlm/output/bb_small
data_dir=${HOME}/lustre/mlm/data/zu

now=$(date +"%Y%m%d_%H%M%S")
start=`date +%s`

mkdir ${out_dir}
#run python script
echo "Running python script"
python -u trainBB.py \
    --output_dir ${out_dir} \
    --data_dir ${data_dir} \
    --model_checkpoint "google/bigbird-roberta-base" \
    --tokenizer_checkpoint "tokenizer/bigbird-st" \
    --evaluation_strategy "steps" \
    --block_size 468 \
    --max_steps 1000 \
    --warmup_steps 10 \
    --learning_rate 1e-4 \
    --weight_decay 0.3 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --fp16 True \
    --eval_steps 100 \
    --save_steps 100 \
    2>&1 | tee ${out_dir}/log

end=`date +%s`
runtime=$(expr $end - $start)
echo "Runtime: ${runtime} seconds" &>> ${out_dir}/log