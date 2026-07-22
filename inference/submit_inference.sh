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