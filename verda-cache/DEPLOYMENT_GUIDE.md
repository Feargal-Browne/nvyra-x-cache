# 🚀 VERDA B300 DEPLOYMENT GUIDE
## Complete Guide: Data Cache Generation on Blackwell Ultra

---

## 📋 Table of Contents

1. [Instance Configuration](#1-instance-configuration)
2. [Cost Estimation](#2-cost-estimation)
3. [SSH Setup & Connection](#3-ssh-setup--connection)
4. [Environment Setup](#4-environment-setup)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Monitoring & Expected Outputs](#6-monitoring--expected-outputs)
7. [Stopping & Billing Management](#7-stopping--billing-management)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Instance Configuration

### Recommended Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| **Instance Type** | **Spot** | €1.055/h vs €4.220/h On-Demand (75% savings) |
| **GPU** | **B300 SXM6 262GB** | Latest Blackwell architecture, sm_120 |
| **Size** | **1B300.30V** | 30 CPUs, 275GB RAM, 262GB VRAM |
| **Contract** | **Pay As You Go** | No commitment, per 10-min billing |
| **Location** | **Finland (FIN-01/02/03)** | Any works, pick available |
| **OS** | **Ubuntu 24.04 + CUDA 13.0 Open + Docker** | Pre-configured for B300 |
| **Storage** | **51 GB OS** + **100GB Block Volume** | For HuggingFace cache |

### Deploy New Instance Settings

```
Instance Type:      Spot
Model:              B300 SXM6 262GB
# GPUs:             1
Size:               1B300.30V (30 CPU, 275GB RAM, 262GB GPU VRAM)
Location:           Finland 1 (FIN-01) or any available
Operating System:   Ubuntu 24.04 + CUDA 13.0 Open + Docker
OS Size:            51 GB
Storage:            Add 100GB Block Volume (for HF cache)
SSH Keys:           [Add your public key]
```

### Startup Script (Optional)

Paste this in the "Startup Script" field:

```bash
#!/bin/bash
# Auto-setup on boot
apt-get update && apt-get install -y htop nvtop tmux
pip install uv
mkdir -p /workspace /data
```

---

## 2. Cost Estimation

### Pricing Breakdown

| Component | Cost | Notes |
|-----------|------|-------|
| B300 Spot Instance | **€1.055/h** | Per GPU |
| Storage (51GB OS) | €0.012/h | Included |
| Block Volume (100GB) | ~€0.01/h | Optional |
| **Total** | **~€1.07/h** | |

### Estimated Processing Time

| Dataset Size | Est. Time | Est. Cost |
|--------------|-----------|-----------|
| 1,000 claims | ~30 min | €0.54 |
| 10,000 claims | ~3 hours | €3.21 |
| 100,000 claims | ~24 hours | €25.68 |
| 1,000,000 claims | ~10 days | €256.80 |

### Cost Control Tips

> ⚠️ **Spot instances can be evicted without warning!**
> - Save checkpoints frequently (built-in)
> - Use the resume feature to continue after eviction
> - Consider On-Demand for critical runs

---

## 3. SSH Setup & Connection

### Step 1: Generate SSH Key (if needed)

```powershell
# On Windows (PowerShell)
ssh-keygen -t ed25519 -C "verda-b300"
cat ~/.ssh/id_ed25519.pub  # Copy this to Verda
```

### Step 2: Add SSH Key to Verda

1. Go to Verda Console → Instance Creation
2. Click "SSH Keys" section
3. Paste your public key

### Step 3: Connect to Instance

After instance is running:

```bash
# Connect (replace with your instance IP)
ssh -i ~/.ssh/id_ed25519 ubuntu@<INSTANCE_IP>

# Or use the hostname provided by Verda
ssh ubuntu@fine-seed-circles-fin-03.verda.cloud
```

### Step 4: Verify GPU

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 580.xx.xx    Driver Version: 580.xx.xx    CUDA Version: 13.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA B300         On   | 00000000:00:1E.0 Off |                    0 |
# | N/A   35C    P0    50W / 700W |      0MiB / 262144MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

---

## 4. Environment Setup

### Step 1: Create Workspace

```bash
# Create directories
mkdir -p /workspace/nvyra-cache
mkdir -p /data/arrow-files
mkdir -p /data/hf-cache

cd /workspace/nvyra-cache
```

### Step 2: Upload Code

From your local machine:

```powershell
# Upload the verda-cache folder
scp -r c:\Users\mabro\OneDrive\Feargal\nvyra-x-code\nvyra-x-models\verda-cache\* ubuntu@<INSTANCE_IP>:/workspace/nvyra-cache/
```

Or clone from git:

```bash
# If code is in a repo
git clone <YOUR_REPO> /workspace/nvyra-cache
```

### Step 3: Install Dependencies

```bash
cd /workspace/nvyra-cache

# Option A: Quick install with uv (recommended)
pip install uv
uv pip install --system -r requirements.txt

# Option B: Standard pip
pip install -r requirements.txt

# Install Flash Attention 3 (CUDA 13.0)
pip install flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch291 \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

### Step 4: Verify Installation

```bash
# Test PyTorch + CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Expected output:
# PyTorch: 2.9.1+cu130
# CUDA: 13.0
# GPU: NVIDIA B300

# Test vLLM
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Expected output:
# vLLM: 0.14.x
```

### Step 5: Upload Data

```bash
# From local machine - upload your Arrow file
scp /path/to/your/data.arrow ubuntu@<INSTANCE_IP>:/data/arrow-files/

# Verify
ls -lh /data/arrow-files/
```

---

## 5. Running the Pipeline

### Option A: Direct Run (Recommended for Testing)

```bash
cd /workspace/nvyra-cache

# Run with tmux (keeps running if SSH disconnects)
tmux new -s cache-build

# Start the pipeline
python3 database_build_verda.py --input-file /data/arrow-files/your_data.arrow

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t cache-build
```

### Option B: Background Run with Logging

```bash
cd /workspace/nvyra-cache

# Run in background with full logging
nohup python3 database_build_verda.py \
    --input-file /data/arrow-files/your_data.arrow \
    > /workspace/logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Check status
tail -f /workspace/logs/run_*.log
```

### Option C: Using Docker

```bash
cd /workspace/nvyra-cache

# Build Docker image
docker build -t nvyra-cache:b300 -f Dockerfile.verda .

# Run with GPU
docker run --gpus all \
    -v /data:/data \
    -v /data/hf-cache:/root/.cache/huggingface \
    nvyra-cache:b300 \
    --input-file /data/arrow-files/your_data.arrow
```

---

## 6. Monitoring & Expected Outputs

### Startup Sequence

```
==============================================================
🚀 VERDA B300 (Blackwell Ultra) Data Cache Generation
==============================================================
   CUDA: 13.0 | PyTorch: 2.9.1 | Compute: sm_120
   Input: /data/arrow-files/your_data.arrow
==============================================================
⚡ Loading Dataset...
⚡ Streaming from /data/arrow-files/your_data.arrow...
⚡ Aggregating & Pruning Context (25k Limit)...
🔍 Connecting to Turso for Resume Check...
   Found 0 previously processed claims.
🔥 Dispatching 50000 items (Fresh Run) to B300...
⚡ Initializing GPU Refinery...
```

### Model Loading Phase

```
⚡ [INIT] Loading Aux Models (BF16 + FA3) on B300...
   ⚡ Loading Qwen3-Reranker-0.6B...
   ⚡ Loading Qwen3-Reranker-8B...
   ⚡ Loading KaLM-Embedding-Gemma3-12B-2511...
   ⚡ Attempting SPLADE with Flash Attention 2 (BFloat16)...
🧠 [INIT] Async VLLM + Nemotron 30B (FP8) [FLASH_ATTN] on B300...
```

### Batch Calibration

```
⚡ [tuning] Starting Smart Binary Search Calibration for B300...
   Testing Batch: 2... ✅ OK
   Testing Batch: 4... ✅ OK
   Testing Batch: 8... ✅ OK
   Testing Batch: 16... ✅ OK
   Testing Batch: 32... ✅ OK
   Testing Batch: 64... ✅ OK
   Testing Batch: 128... ✅ OK
   Testing Batch: 256... ❌ OOM
   🔍 Refining between 128 and 256...
   Testing Batch: 192... ✅ OK
   Testing Batch: 224... ❌ OOM
   Testing Batch: 208... ✅ OK
⚡ [tuning] Exact Max: 208. Optimized Batch Size: 197
```

### Processing Metrics (Periodic)

```json
{"metric": "pipeline_status", "uptime_sec": 300, "throughput_tps": "5.23", "processed": 1570, "saved": {"turso": 1570, "qdrant": 1570, "s3": 1570}}
{"metric": "pipeline_status", "uptime_sec": 600, "throughput_tps": "5.18", "processed": 3109, "saved": {"turso": 3109, "qdrant": 3109, "s3": 3109}}
{"metric": "pipeline_status", "uptime_sec": 900, "throughput_tps": "5.21", "processed": 4693, "saved": {"turso": 4693, "qdrant": 4693, "s3": 4693}}
```

### Completion

```
🚀 Starting B300 GPU Monolith...
... [processing logs] ...
👋 Job Fully Complete.
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop for prettier display
nvtop
```

Expected during processing:
- GPU Memory: ~200-250GB used (of 262GB)
- GPU Util: 70-95%
- Power: 400-600W

---

## 7. Stopping & Billing Management

### Graceful Stop (Keep Resume Capability)

```bash
# Find the process
ps aux | grep database_build

# Send interrupt signal (graceful)
kill -SIGINT <PID>

# Or press Ctrl+C if running in foreground
```

The pipeline saves checkpoints to Turso, so you can resume:

```bash
# Resume - will skip already processed claims
python3 database_build_verda.py --input-file /data/arrow-files/your_data.arrow

# Expected output:
# Found 5000 previously processed claims.
# ⏩ SKIPPING 5000 items (already in DB).
# 🔥 Dispatching 45000 Remaining items to B300...
```

### Stop Instance (Stop Billing)

#### Via Verda Console:

1. Go to Verda Console → Instances
2. Find your instance
3. Click **Stop** button
4. Confirm

#### Via CLI (if available):

```bash
verda instance stop <INSTANCE_ID>
```

### Terminate Instance (End All Billing)

> ⚠️ **WARNING**: This deletes the instance. Make sure to:
> 1. Download any local data you need
> 2. Verify vectors are saved to Qdrant (Contabo VM)
> 3. Verify metadata is saved to Turso

```bash
# Backup before terminating
scp -r ubuntu@<INSTANCE_IP>:/workspace/logs ./backup_logs/

# Then in Verda Console:
# Click "Delete" on the instance
```

### Billing Summary

| Action | Billing Status |
|--------|----------------|
| Instance Running | **Charging** (~€1.07/h) |
| Instance Stopped | **Charging** for storage only (~€0.02/h) |
| Instance Deleted | **No Charges** |
| Spot Eviction | Automatic stop, no data loss if using checkpoints |

---

## 8. Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If error, driver may need reinstall
sudo apt-get install --reinstall nvidia-driver-580
```

#### CUDA Version Mismatch

```bash
# Verify CUDA
nvcc --version
# Should show: Cuda compilation tools, release 13.0

# If wrong version, reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130
```

#### Out of Memory (OOM)

```bash
# Reduce batch size - edit database_build_verda.py:
pipeline_batch_size = 64  # Reduce from 128

# Or reduce vLLM memory:
vllm_gpu_utilisation = 0.40  # Reduce from 0.50
```

#### Spot Instance Evicted

```
# Just restart the instance and resume
python3 database_build_verda.py --input-file /data/arrow-files/your_data.arrow
# Will automatically skip already-processed claims
```

#### Connection to Storage Failed

```bash
# Test Turso connection
python3 -c "import libsql_experimental as libsql; db = libsql.connect(database='https://ai-metadata-cache-f-b.aws-eu-west-1.turso.io', auth_token='...'); print('Turso OK')"

# Test Qdrant (Contabo VM)
python3 -c "from qdrant_client import QdrantClient; qc = QdrantClient(url='http://95.111.232.85:6333'); print(qc.get_collections())"

# Test S3 (Backblaze)
python3 -c "import boto3; s3 = boto3.client('s3', endpoint_url='https://s3.eu-central-003.backblazeb2.com', aws_access_key_id='00356bc3d6937610000000004', aws_secret_access_key='K0036GxH+hhmmADw9yh8aspgXhvu6fo'); print(s3.list_buckets())"
```

---

## 📊 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    VERDA B300 QUICK REF                     │
├─────────────────────────────────────────────────────────────┤
│ CONNECT:    ssh ubuntu@<IP>                                 │
│ START:      python3 database_build_verda.py --input-file X  │
│ MONITOR:    nvidia-smi / nvtop / tail -f logs/*.log         │
│ STOP:       Ctrl+C (graceful) or kill -SIGINT <PID>         │
│ RESUME:     Same command - auto-skips processed claims      │
│ COST:       €1.07/h (Spot) / €4.22/h (On-Demand)            │
│ EVICTION:   No panic - just restart and resume              │
├─────────────────────────────────────────────────────────────┤
│ STORAGE BACKENDS:                                           │
│   Turso:  ai-metadata-cache-f-b.aws-eu-west-1.turso.io     │
│   Qdrant: 95.111.232.85:6333 (Contabo VM)                  │
│   S3:     s3.eu-central-003.backblazeb2.com                │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Pre-Flight Checklist

Before running, verify:

- [ ] Instance is running (check Verda Console)
- [ ] SSH connection works
- [ ] `nvidia-smi` shows B300 GPU
- [ ] Python + PyTorch + CUDA 13.0 installed
- [ ] Arrow input file uploaded to `/data/arrow-files/`
- [ ] Using `tmux` or `nohup` (survives SSH disconnect)
- [ ] Storage backends reachable (Turso, Qdrant, S3)

**Ready to go! 🚀**
