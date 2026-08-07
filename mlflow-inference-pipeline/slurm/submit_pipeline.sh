#!/bin/bash
#SBATCH --job-name=mlflow-inference
#SBATCH --account=bmj-842-02
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Usage:
#   sbatch submit_pipeline.sh descriptions Qwen/Qwen3-8B dataset/original_code
#   sbatch submit_pipeline.sh code Qwen/Qwen3-8B outputs/generated_descriptions/<date>-<model>

TASK=$1        # "descriptions" or "code"
MODEL=$2       # e.g. Qwen/Qwen3-8B
INPUT_PATH=$3  # input-dir (descriptions) or desc-input-dir (code)

mkdir -p logs

module load python/3.11 2>/dev/null || true
source venv/bin/activate

if [ "$TASK" == "descriptions" ]; then
    python run_pipeline.py descriptions --model "$MODEL" --input-dir "$INPUT_PATH"
elif [ "$TASK" == "code" ]; then
    python run_pipeline.py code --model "$MODEL" --desc-input-dir "$INPUT_PATH"
else
    echo "Unknown task: $TASK (expected 'descriptions' or 'code')"
    exit 1
fi
