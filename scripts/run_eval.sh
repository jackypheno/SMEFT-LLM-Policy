#!/bin/bash
#SBATCH --job-name=eval-policy
#SBATCH --output=/home/kumarj/links/scratch/eval_policy_%j.out        # %j = job ID
#SBATCH --error=/home/kumarj/links/scratch/eval_policy_%j.err
#SBATCH --time=04:00:00                   # hh:mm:ss
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --account=def-london

# Load modules if needed (comment if not required)

start_time=$(date +%s)
echo "Job started at: $(date)"

module load gcc/12.3 arrow cuda/12.6 python/3.10
#module load gcc/12.3 arrow cuda/12.6 python/3.11.5

cd /home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly

source ENV/bin/activate

python -m src.generate_mask

end_time=$(date +%s)
echo "Job ended at: $(date)"

elapsed=$(( end_time - start_time ))
echo "Elapsed time: $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s"
