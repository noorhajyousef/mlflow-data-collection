#!/bin/bash
# submit_inference.sh -- SLURM array job for the full inference run on Fir.
# Matches the Task Design Proposal pipeline sketch: "SLURM array job; loads
# one model on one 40GB MIG slice, generates completions for all instances
# in both modes."
#
# Submit ONCE (the array covers both models):
#   sbatch submit_inference.sh
#
# Array index 0 = qwen, 1 = llama. If a task is killed or times out, just
# resubmit that index; run_inference.py resumes from its output file:
#   sbatch --array=1 submit_inference.sh     # rerun llama only

#SBATCH --job-name=mlflow-inference
#SBATCH --account=def-abdellatif        # TODO(Noor): confirm with 'sacctmgr show user noorysf9 withassoc format=account%30'
#SBATCH --array=0-1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

MODELS=(qwen llama)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

module load python/3.11 gcc arrow/22.0.0
source ~/mlflow_project/env/bin/activate

# Weights already cached on scratch; compute nodes have no internet.
export HF_HOME=/scratch/noorysf9/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ~/mlflow_project
mkdir -p outputs logs

python run_inference.py \
    --model "$MODEL" \
    --instances instances.jsonl \
    --output "outputs/${MODEL}.jsonl"

echo "Array task $SLURM_ARRAY_TASK_ID finished for model: $MODEL"
