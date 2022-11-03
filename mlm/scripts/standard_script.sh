#!/bin/bash
#PBS -P CSCI1335
#PBS -N gpt_final_zu
#PBS -q gpu_1
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -l walltime=12:00:00
#PBS -e /home/jvanschalkwyk/lustre/mlm/standard_err
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

out_dir=${HOME}/lustre/mlm/output/gpt2_final_zu_false
data_dir=${HOME}/lustre/mlm/data/zu

now=$(date +"%Y%m%d_%H%M%S")
start=`date +%s`

mkdir ${out_dir}
#run python script
echo "Running python script"
python -u train.py \
    --output_dir ${out_dir} \
    --data_dir ${data_dir} \
    --model_checkpoint "gpt2" \
    --tokenizer_checkpoint "tokenizer/gpt2" \
    --evaluation_strategy "steps" \
    --block_size 128 \
    --max_steps 200000 \
    --warmup_steps 2000 \
    --learning_rate 1e-4 \
    --weight_decay 0.3 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --fp16 True \
    --eval_steps 1000 \
    --save_steps 1000 \
    2>&1 | tee ${out_dir}/log1

end=`date +%s`
runtime=$(expr $end - $start)
echo "Runtime: ${runtime} seconds" &>> ${out_dir}/log1
