#!/bin/bash
#SBATCH --job-name=job_wgpu
#SBATCH --account=ds_ga_1011-2024fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay /scratch/zf2345/overlay-25GB-500K.ext3:rw /scratch/zf2345/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate hw3-nlp
python3 main.py --train --eval --debug train
"
