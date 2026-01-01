#%%writefile sota_project/pipeline.py
import modal
import asyncio
import uuid
import os
from typing import List
from pydantic import BaseModel

# --- CONFIGURATION ---
APP_NAME = "sota-h200-final"

# --- INFRASTRUCTURE ---
# Queue buffers the Rust blast
job_queue = modal.Queue.from_name("sota_job_queue", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

# --- CUSTOM IMAGE (CUDA 12.8 | FA3 | Torch 2.9.1) ---
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .pip_install("uv")
    .run_commands(
        "uv venv /root/env",
        "uv pip install --system --upgrade pip",
        "uv pip install --system pydantic fastapi uvicorn pandas pyarrow huggingface_hub transformers>=4.48.0",
        "uv pip install --system 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128",
        "uv pip install --system flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 --extra-index-url https://download.pytorch.org/whl/cu128",
        "uv pip install --system vllm>=0.13.0 --no-build-isolation"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_FLASH_ATTN_VERSION": "3",
        "NCCL_P2P_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": "0"
    })
    .run_function(download_model, secrets=[hf_secret])
)

app = modal.App(APP_NAME)

class IngestBatch(BaseModel):
    items: List[dict]

# --- 1. INGEST API ---
@app.function(
    image=modal.Image.debian_slim().pip_install("pydantic"),
    max_containers=20
)
@modal.fastapi_endpoint(method="POST")
async def ingest(batch: IngestBatch):
    # Rust sends 256 items at once. We verify and push.
    await job_queue.put_many(batch.items)
    return {"status": "ok"}

# --- 2. SINGLE H200 WORKER ---
@app.cls(
    image=image,
    gpu="H200",
    secrets=[hf_secret],
    max_containers=1,            # STRICT: Single Worker
    scaledown_window=2,          # STRICT: Kill 2s after done
    enable_memory_snapshot=True, # FAST START: Hibernate model
    timeout=900
)
class H200Worker:
    @modal.enter()
    def setup(self):
        """Runs once at build time to load model."""
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        print("📸 SNAPSHOT: Loading Nemotron-3...")
        self.llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
            max_model_len=8192,
            kv_cache_dtype="fp8",
            max_num_seqs=256,
            disable_log_stats=True,
            gpu_memory_utilization=0.95
        ))
        self.sampling_params = SamplingParams(temperature=0.1, max_tokens=512)
        print("📸 SNAPSHOT: Hibernating.")

    @modal.method()
    async def process_until_empty(self):
        print("⚡ WAKEUP: Processing...")
        while True:
            try:
                # If queue is empty for 1s, we assume job is done.
                batch = await job_queue.get_many(256, block=True, timeout=1.0)
            except Exception:
                print("🛑 Queue empty. Shutting down.")
                break
            
            if not batch: break
            
            # Fire inference
            tasks = [
                self.llm.generate(
                    f"<extra_id_0>System\nExtract\n<extra_id_1>User\n{item['text']}\n<extra_id_1>Assistant\n",
                    self.sampling_params, 
                    f"{item['id']}-{uuid.uuid4()}"
                ) for item in batch
            ]
            await asyncio.gather(*tasks)

@app.local_entrypoint()
def start():
    # Spawns the worker detached. It will wake up, consume, and die.
    H200Worker().process_until_empty.spawn()