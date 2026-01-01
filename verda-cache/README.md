# Verda B300 Data Cache Generation

**GPU**: NVIDIA B300 (Blackwell Ultra) - sm_120  
**CUDA**: 13.0  
**PyTorch**: 2.9.1 (cu130)  
**Base Image**: Ubuntu 24.04 + CUDA 13.0 Open + Docker

## Files

| File | Description |
|------|-------------|
| `database_build_verda.py` | Main processing script (all Modal features preserved) |
| `Dockerfile.verda` | Docker build file for CUDA 13.0 |
| `requirements.txt` | Python dependencies |
| `run_verda.sh` | SLURM submission script |
| `build.sh` | Build Docker/Singularity images |

## Features Preserved from Original

- ✅ Dual reranker (Qwen3-0.6B + Qwen3-8B)
- ✅ Dense embeddings (KaLM-Gemma3-12B)
- ✅ Sparse embeddings (SPLADE v3)
- ✅ vLLM async inference (Nemotron-30B FP8)
- ✅ Triple storage (Turso + Qdrant + S3)
- ✅ Contabo VM Qdrant vectors preserved
- ✅ Resume from checkpoint
- ✅ Deduplication cache
- ✅ Smart batch calibration

## Storage Backends

| Backend | Location | Purpose |
|---------|----------|---------|
| **Turso** | `ai-metadata-cache-f-b.aws-eu-west-1.turso.io` | Claim metadata |
| **Qdrant** | `http://95.111.232.85:6333` (Contabo VM) | Vector storage |
| **S3** | `s3.eu-central-003.backblazeb2.com` | Compressed JSON |

## Quick Start

### On Verda HPC

```bash
# 1. Build Singularity image (one-time)
./build.sh

# 2. Submit job
sbatch run_verda.sh /path/to/data.arrow
```

### Local Docker (for testing)

```bash
# Build
docker build -t nvyra-cache-verda:b300-cuda13 -f Dockerfile.verda .

# Run
docker run --gpus all -v /data:/data nvyra-cache-verda:b300-cuda13 \
    --input-file /data/test.arrow
```

## SLURM Configuration

Edit `run_verda.sh` to set:
- `--partition=<YOUR_B300_PARTITION>` - Your B300 GPU partition name
- `--account=<YOUR_ACCOUNT>` - Your compute account

## Key Differences from Modal Version

| Aspect | Modal (Original) | Verda (This) |
|--------|-----------------|--------------|
| Orchestration | `modal.App` | Standalone Python |
| GPU Spec | `gpu="H200"` | Runtime B300 detection |
| Image Build | `modal.Image` | Dockerfile |
| Volumes | `modal.Volume` | Local/mounted paths |
| Scheduling | Modal cloud | SLURM |
