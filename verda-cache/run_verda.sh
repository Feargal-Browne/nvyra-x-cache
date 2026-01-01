#!/bin/bash
# ==============================================================================
# Verda HPC SLURM Submission Script
# B300 (Blackwell Ultra) GPU Job
# ==============================================================================

#SBATCH --job-name=nvyra-cache-build
#SBATCH --output=logs/cache_build_%j.out
#SBATCH --error=logs/cache_build_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# ==============================================================================
# NOTE: Update these based on your Verda cluster configuration
# ==============================================================================
# #SBATCH --partition=<YOUR_B300_PARTITION>  # e.g., gpu, blackwell, b300
# #SBATCH --account=<YOUR_ACCOUNT>

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "VERDA B300 Data Cache Generation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# ==============================================================================
# Load Modules (adjust based on Verda's module system)
# ==============================================================================
module purge
module load cuda/13.0
module load python/3.12

# Verify GPU
echo "GPU Information:"
nvidia-smi

# ==============================================================================
# Option 1: Run with Singularity Container
# ==============================================================================
# First convert Docker image to Singularity:
# singularity build database_build.sif docker://your-registry/database-build-verda:latest

# singularity exec --nv \
#     --bind /data:/data \
#     database_build.sif \
#     python3 database_build_verda.py --input-file "$1"

# ==============================================================================
# Option 2: Run with Python directly (if deps installed)
# ==============================================================================
# Activate virtual environment if using one
# source /path/to/venv/bin/activate

# Set environment for B300
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=12.0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run the script
python3 database_build_verda.py --input-file "$1"

echo "=============================================="
echo "End Time: $(date)"
echo "=============================================="
