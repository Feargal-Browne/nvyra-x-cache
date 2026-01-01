# ------------------------------------------------------------------------------
#  CONFIDENTIAL AND PROPRIETARY
#  Copyright (c) 2025-2026 nvyra-x. All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of
#  nvyra-x. The intellectual and technical concepts contained herein are
#  proprietary to nvyra-x and may be covered by Irish and Foreign Patents,
#  patents in process, and are protected by trade secret or copyright law.
#  Dissemination of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained from nvyra-x.
# ------------------------------------------------------------------------------

# %%writefile y.py
import sys
import os
import time
import json
import uuid
import asyncio
import modal
import re
import queue
import hashlib
from typing import List, Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Configuration
APP_NAME = "data-cache-generation"
data_vol = modal.Volume.from_name("rag-harvest-storage-prod", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# SECRET MAPPING
my_secrets = [
    modal.Secret.from_name("huggingface-secret"), 
    modal.Secret.from_name("turso-api-new")
]

# TUNING
vllm_gpu_utalisation = 0.50  
pipeline_batch_size = 128        
max_gpus = 1              

# ENV
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN" 
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

# Image Definition 

def download_artifacts():
    print("⚡ [BUILD] Parallel Fetching...")
    from transformers import AutoTokenizer
    models = [
        "Qwen/Qwen3-Reranker-0.6B",
        "Qwen/Qwen3-Reranker-8B", 
        "tencent/KaLM-Embedding-Gemma3-12B-2511",
        "naver/splade-v3", 
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    ]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        ex.map(lambda m: AutoTokenizer.from_pretrained(m, trust_remote_code=True), models)

# GPU Image 
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "libzstd-dev", "build-essential", "ninja-build")
    .pip_install("uv")
    .run_commands(
        "uv venv .venv",
        "uv pip install --system --upgrade setuptools",
        "uv pip install --system --index-url https://download.pytorch.org/whl/test/cu126 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0",
        "uv pip install --system anthropic==0.71.0 numba==0.61.2 lark==1.2.2 compressed-tensors==0.12.2 llguidance==1.3.0 xgrammar==0.1.27 transformers==4.57.3 accelerate>=1.2.0 outlines>=0.1.0 fastembed aiohttp boto3 pyarrow zstandard bitarray qdrant-client uvloop polars bitsandbytes>=0.45.0 hf_transfer datasketch ninja blake3 cachetools cbor2 depyf fastapi gguf ijson lm-format-enforcer mcp mistral_common model-hosting-container-standards msgspec openai openai-harmony opencv-python-headless partial-json-parser prometheus_client prometheus-fastapi-instrumentator py-cpuinfo pybase64 python-json-logger pyzmq ray sentencepiece setproctitle tiktoken watchfiles libsql-experimental rank-bm25",
        "uv pip install --system --no-deps vllm==0.13.0",
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu126_torch290 --extra-index-url https://download.pytorch.org/whl/cu126",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_artifacts, secrets=my_secrets, volumes={"/root/.cache/huggingface": hf_cache_vol})
)

# CPU image 
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("libsql-experimental", "boto3", "qdrant-client", "polars", "pyarrow", "rank-bm25", "datasketch", "transformers", "hf_transfer")
    .run_commands("python -m spacy download en_core_web_sm || true")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME, secrets=my_secrets)

# Logging Metrics

@dataclass
class PipelineMetrics:
    start_time: float = field(default_factory=time.time)
    items_processed: int = 0
    items_saved_turso: int = 0
    items_saved_qdrant: int = 0
    items_saved_s3: int = 0
    errors_db: int = 0
    
    def report(self):
        duration = time.time() - self.start_time
        tps = self.items_processed / duration if duration > 0 else 0
        print(json.dumps({
            "metric": "pipeline_status",
            "uptime_sec": int(duration),
            "throughput_tps": f"{tps:.2f}",
            "processed": self.items_processed,
            "saved": {"turso": self.items_saved_turso, "qdrant": self.items_saved_qdrant, "s3": self.items_saved_s3}
        }))

#Dataset Loader (CPU)
@app.cls(image=cpu_image, volumes={"/data": data_vol}, timeout=1200)
class DatasetLoader:
    @modal.method()
    def process_and_rank(self, input_file: str) -> List[Dict]:
        import polars as pl
        import pyarrow as pa
        from rank_bm25 import BM25Okapi
        from transformers import AutoTokenizer
        from datasketch import MinHash
        
        tok_lite = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True)
        tok_heavy = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-8B", trust_remote_code=True)

        print(f"⚡ Streaming from {input_file}...")
        try:
            with open(input_file, 'rb') as f:
                reader = pa.ipc.open_stream(f)
                pa_table = reader.read_all()
            df = pl.from_arrow(pa_table)
        except Exception as e:
            print(f"❌ Failed to read Arrow Stream: {e}")
            return []
        
        print("⚡ Aggregating & Pruning Context (25k Limit)...")
        df = df.group_by("claim_id").agg([
            pl.col("claim_text").first(),
            pl.col("raw_doc_text").alias("docs"),
            pl.col("url").alias("urls"),
            pl.col("title").alias("titles"),
            pl.col("source").alias("sources")
        ])
        
        data = df.to_dicts()
        processed_items = []
        
        for row in data:
            claim = row['claim_text']
            docs = row['docs']
            unique_docs = []
            hashes = []
            for doc in docs:
                m = MinHash(num_perm=128)
                for word in doc.split(): m.update(word.encode('utf8'))
                is_duplicate = False
                for h in hashes:
                    if m.jaccard(h) > 0.85: 
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_docs.append(doc)
                    hashes.append(m)
            
            if len(unique_docs) > 0:
                tokenized_claim = claim.lower().split()
                tokenized_docs = [d.lower().split() for d in unique_docs]
                bm25 = BM25Okapi(tokenized_docs)
                scores = bm25.get_scores(tokenized_claim)
                scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
                
                selected_docs = []
                current_len = 0
                for d, s in scored_docs:
                    if current_len + len(d) < 15000: 
                        selected_docs.append(d)
                        current_len += len(d)
                    else: break
                combined_docs = "\n\n".join(selected_docs)
            else:
                combined_docs = ""

            lite_p = f"Instruct: Retrieve relevant context\nQuery: {claim}\nDoc: {combined_docs[:512]}"
            heavy_p = f"<Instruct>: Identify contradictions and supporting evidence\n<Query>: {claim}\n<Document>: {combined_docs}"
            
            lite_ids = tok_lite.encode(lite_p, truncation=True, max_length=1024)
            heavy_ids = tok_heavy.encode(heavy_p, truncation=True, max_length=4096)
            prefix = heavy_ids[:64]
            lcp_hash = hash(tuple(prefix))
            
            processed_items.append({
                "claim_id": row['claim_id'],
                "claim_text": claim,
                "doc_text": combined_docs, 
                "doc_metadata": {
                    "urls": row['urls'], 
                    "titles": row['titles'], 
                    "sources": row['sources']
                },
                "lite_ids": lite_ids,
                "heavy_ids": heavy_ids,
                "lcp_hash": lcp_hash
            })
            
        processed_items.sort(key=lambda x: x['lcp_hash'])
        return processed_items

# Database Scanner
@app.cls(image=cpu_image, secrets=my_secrets)
class DBScanner:
    @modal.method()
    def get_existing_ids(self) -> List[str]:
        import libsql_experimental as libsql
        print("🔍 Connecting to Turso for Resume Check...")
        try:
            turso_url = "https://ai-metadata-cache-f-b.aws-eu-west-1.turso.io"
            turso_token = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3NjYzNDE4NzEsImlkIjoiYmYwODMzM2MtNTZlMS00ZDJhLWIwYmItMGUzOTMyODI0Y2FlIiwicmlkIjoiMjBmOGYyNjgtODkzYS00NTk5LWI0NWYtMDc3M2MxOGYwNjZiIn0.U-A2yG0WcrG1gikhyNrreLm9cDqlQstgiT9IW9mtgM111xNKjEnoEohOnWY9uNXD2kGpe-tHfb54b_hHCXvEBw"
            db = libsql.connect(database=turso_url, auth_token=turso_token)
            res = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='claim_metadata'")
            if not res.fetchone(): return []
            rows = db.execute("SELECT claim_id FROM claim_metadata").fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            print(f"⚠️ Failed to fetch existing IDs: {e}")
            return []

# Modal Paramaters
@app.cls(
    image=gpu_image, 
    gpu="H200", 
    volumes={"/data": data_vol, "/root/.cache/huggingface": hf_cache_vol}, 
    max_containers=max_gpus,
    timeout=5400
)
class GodModeRefinery:
    
    @modal.enter()
    def setup(self):
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForMaskedLM
        from vllm import SamplingParams
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        import boto3
        from botocore.config import Config
        import zstandard
        import libsql_experimental as libsql
        from qdrant_client import QdrantClient
        
        self.device = "cuda"
        self.dedup_cache = OrderedDict()
        self.dedup_limit = 50000
        self.aux_stream = torch.cuda.Stream()
        self.metrics = PipelineMetrics()
        
        # --- STORAGE INIT (Inside Monolith) ---
        self.cctx = zstandard.ZstdCompressor(level=3)
        self.s3 = boto3.client(
            's3', 
            endpoint_url="https://s3.eu-central-003.backblazeb2.com", 
            aws_access_key_id="00356bc3d6937610000000004", 
            aws_secret_access_key="K0036GxH+hhmmADw9yh8aspgXhvu6fo",
            config=Config(max_pool_connections=100) # 🔥 Bottleneck fix
        )
        self.qc = QdrantClient(url="http://95.111.232.85:6333")
        turso_url = "https://ai-metadata-cache-f-b.aws-eu-west-1.turso.io"
        turso_token = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3NjYzNDE4NzEsImlkIjoiYmYwODMzM2MtNTZlMS00ZDJhLWIwYmItMGUzOTMyODI0Y2FlIiwicmlkIjoiMjBmOGYyNjgtODkzYS00NTk5LWI0NWYtMDc3M2MxOGYwNjZiIn0.U-A2yG0WcrG1gikhyNrreLm9cDqlQstgiT9IW9mtgM111xNKjEnoEohOnWY9uNXD2kGpe-tHfb54b_hHCXvEBw"
        self.db = libsql.connect(database=turso_url, auth_token=turso_token)
        
        # Ensure Schema
        self.db.execute("CREATE TABLE IF NOT EXISTS claim_metadata (id TEXT PRIMARY KEY, claim_id TEXT, verdict TEXT, falsity_score REAL, lite_score REAL, heavy_score REAL, s3_key TEXT, source_urls TEXT, source_titles TEXT, source_publishers TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_claim_id ON claim_metadata(claim_id)")
        self.db.commit()
        
        # --- MODEL LOADING ---
        print("⚡ [INIT] Loading Aux Models (BF16 + FA3)...")
        def load_fast(model, cls, **kwargs):
            return cls.from_pretrained(model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_3", **kwargs).to("cuda:0").eval()

        def load_robust_splade(model_name, cls, **kwargs):
            try:
                print(f"   ⚡ Attempting SPLADE with Flash Attention 3 (BFloat16)...")
                return cls.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **kwargs).to("cuda:0").eval()
            except Exception: pass
            try:
                print(f"   ⚡ Attempting SPLADE with SDPA (BFloat16)...")
                return cls.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs).to("cuda:0").eval()
            except Exception: pass
            print(f"   🐢 Falling back to SPLADE Eager Mode (BFloat16)...")
            return cls.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager", **kwargs).to("cuda:0").eval()

        self.model_lite = load_fast("Qwen/Qwen3-Reranker-0.6B", AutoModelForCausalLM, trust_remote_code=True)
        self.model_heavy = load_fast("Qwen/Qwen3-Reranker-8B", AutoModelForCausalLM, trust_remote_code=True)
        self.model_dense = load_fast("tencent/KaLM-Embedding-Gemma3-12B-2511", AutoModel, trust_remote_code=True)
        self.model_sparse = load_robust_splade("naver/splade-v3", AutoModelForMaskedLM, trust_remote_code=True)
        
        self.tok_sparse = AutoTokenizer.from_pretrained("naver/splade-v3", trust_remote_code=True)
        self.tok_main = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", trust_remote_code=True)
        self.tok_dense = AutoTokenizer.from_pretrained("tencent/KaLM-Embedding-Gemma3-12B-2511", trust_remote_code=True)

        self.yes_id = 7866 
        self.no_id = 1489 

        print(f"🧠 [INIT] Async VLLM + Nemotron 30B (FP8) [FLASH_ATTN] on H200...")
        engine_args = AsyncEngineArgs(
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
            trust_remote_code=True,
            gpu_memory_utilization=vllm_gpu_utalisation,
            max_num_seqs=2048, 
            max_num_batched_tokens=8192, 
            kv_cache_dtype="fp8",
            enforce_eager=True, 
            disable_log_stats=False
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(temperature=0.8, max_tokens=2048)
        
        self.optimal_batch_size = self._calibrate_batch_size_smart()

    def _calibrate_batch_size_smart(self):
        import torch
        print("⚡ [tuning] Starting Smart Binary Search Calibration...")
        low, high, current, max_limit = 2, None, 2, 256
        dummy_ids = [[1] * 1024] * max_limit 
        
        with torch.cuda.stream(self.aux_stream):
            with torch.inference_mode():
                while current <= max_limit:
                    try:
                        print(f"   Testing Batch: {current}...", end="", flush=True)
                        batch = dummy_ids[:current]
                        input_tensor = torch.tensor(batch, device=self.device)
                        attn_mask = (input_tensor != 0).long()
                        _ = self.model_heavy(input_ids=input_tensor, attention_mask=attn_mask)
                        del input_tensor, attn_mask
                        print(" ✅ OK")
                        low = current
                        current *= 2
                    except RuntimeError:
                        print(" ❌ OOM")
                        high = current
                        torch.cuda.empty_cache()
                        break
                
                if high is None: high = max_limit 
                print(f"   🔍 Refining between {low} and {high}...")
                final_safe = low
                while low <= high:
                    mid = (low + high) // 2
                    if mid == final_safe: break 
                    try:
                        print(f"   Testing Batch: {mid}...", end="", flush=True)
                        batch = dummy_ids[:mid]
                        input_tensor = torch.tensor(batch, device=self.device)
                        attn_mask = (input_tensor != 0).long()
                        _ = self.model_heavy(input_ids=input_tensor, attention_mask=attn_mask)
                        del input_tensor, attn_mask
                        print(" ✅ OK")
                        final_safe = mid
                        low = mid + 1
                    except RuntimeError:
                        print(" ❌ OOM")
                        torch.cuda.empty_cache()
                        high = mid - 1
        
        optimal = int(final_safe * 0.95)
        print(f"⚡ [tuning] Exact Max: {final_safe}. Optimized Batch Size: {optimal}")
        return optimal

    def _run_aux_model(self, model, input_ids: List[List[int]], task_type: str, texts: List[str] = None):
        import torch
        import numpy as np
        MICRO_BATCH_SIZE = self.optimal_batch_size

        with torch.cuda.stream(self.aux_stream):
            with torch.inference_mode():
                if task_type == "sparse":
                    CH_SIZE, STRIDE = 512, 256
                    batch_indices, batch_values = [], []
                    for text in texts:
                        tokens = self.tok_sparse(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=8192).to(self.device)
                        total_len = tokens['input_ids'].shape[1]
                        chunks = [tokens] if total_len <= CH_SIZE else [
                            {k: v[:, i:min(i + CH_SIZE, total_len)] for k, v in tokens.items()}
                            for i in range(0, total_len, STRIDE)
                        ]
                        chunk_vecs = []
                        for c in chunks:
                            out = model(**c)
                            val = torch.max(torch.log(1 + torch.relu(out.logits)) * c['attention_mask'].unsqueeze(-1), dim=1).values.squeeze()
                            chunk_vecs.append(val)
                        final_vec = torch.stack(chunk_vecs).max(dim=0).values if len(chunk_vecs) > 1 else chunk_vecs[0]
                        indices = final_vec.nonzero().squeeze().cpu().tolist()
                        values = final_vec[indices].cpu().tolist()
                        batch_indices.append([indices] if isinstance(indices, int) else indices)
                        batch_values.append([values] if isinstance(values, float) else values)
                    return batch_indices, batch_values

                results = []
                for i in range(0, len(input_ids), MICRO_BATCH_SIZE):
                    batch_slice = [x[:1024] for x in input_ids[i : i + MICRO_BATCH_SIZE]]
                    max_len = max(len(x) for x in batch_slice)
                    padded = [x + [0]*(max_len - len(x)) for x in batch_slice]
                    input_tensor = torch.tensor(padded, device=self.device)
                    attn_mask = (input_tensor != 0).long()

                    if task_type == "rerank":
                        logits = model(input_ids=input_tensor, attention_mask=attn_mask).logits[:, -1, :]
                        scores = torch.softmax(logits[:, [self.no_id, self.yes_id]], dim=1)[:, 1].cpu().float().numpy()
                        results.extend(scores)
                    elif task_type == "dense":
                        out = model(input_ids=input_tensor, attention_mask=attn_mask)
                        last_idx = attn_mask.sum(1) - 1
                        vecs = out[0][torch.arange(len(batch_slice)), last_idx][:, :1024]
                        binarized = (torch.nn.functional.normalize(vecs, p=2, dim=1) > 0).int().cpu().tolist()
                        results.extend(binarized)
                    
                    del input_tensor, attn_mask
                    torch.cuda.empty_cache() 
                return np.array(results) if task_type == "rerank" else results

    def _check_dedup(self, claim_text: str) -> Optional[Dict]:
        h = hashlib.md5(claim_text.encode()).hexdigest()
        if h in self.dedup_cache:
            self.dedup_cache.move_to_end(h)
            return self.dedup_cache[h]
        return None

    def _update_dedup(self, claim_text: str, result: Dict):
        h = hashlib.md5(claim_text.encode()).hexdigest()
        self.dedup_cache[h] = result
        if len(self.dedup_cache) > self.dedup_limit:
            self.dedup_cache.popitem(last=False)

    async def _generate_async(self, req_id: str, prompt: str):
        try:
            results_generator = self.llm.generate(prompt, self.sampling_params, req_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            return final_output.outputs[0].text
        except Exception: return None

    # Save Logic
    def _save_batch(self, flat_results):
        if not flat_results: return
        
        # Lazy imports for safe saving
        from qdrant_client.models import PointStruct, SparseVector
        
        def _up(x):
            exclude = ['dense_vec', 'sparse_idx', 'sparse_val', 'lite_score', 'heavy_score', 'doc_metadata']
            payload = {k: v for k, v in x.items() if k not in exclude}
            k = f"v30/{x['claim_id']}/{x['record_uuid']}.zst"
            self.s3.put_object(Bucket="ai-text-cache", Key=k, Body=self.cctx.compress(json.dumps(payload).encode()))
            return k
        
        with ThreadPoolExecutor(64) as ex: keys = list(ex.map(_up, flat_results))
        self.metrics.items_saved_s3 += len(keys)

        try:
            self.db.execute("BEGIN TRANSACTION")
            for i, x in enumerate(flat_results):
                self.db.execute(
                    "INSERT OR IGNORE INTO claim_metadata (id, claim_id, verdict, falsity_score, lite_score, heavy_score, s3_key, source_urls, source_titles, source_publishers) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (x['record_uuid'], x['claim_id'], x['analysis']['verdict'], x['analysis']['falsity_score'], x.get('lite_score', 0.0), x.get('heavy_score', 0.0), keys[i], json.dumps(x['doc_metadata']['urls']), json.dumps(x['doc_metadata']['titles']), json.dumps(x['doc_metadata']['sources']))
                )
            self.db.commit()
            self.metrics.items_saved_turso += len(flat_results)
        except Exception as e:
            print(f"⚠️ Turso Error: {e}")
            self.db.rollback()
            self.metrics.errors_db += 1

        pts = []
        for i, x in enumerate(flat_results):
            pts.append(PointStruct(
                id=x['record_uuid'], 
                vector={"dense": x['dense_vec'], "sparse": SparseVector(indices=x['sparse_idx'], values=x['sparse_val'])}, 
                payload={"cid": x['claim_id']}
            ))
        self.qc.upsert("diamond_v30", points=pts, wait=False)
        self.metrics.items_saved_qdrant += len(pts)
        self.metrics.items_processed += len(flat_results)
        self.metrics.report()

    @modal.method()
    async def process_batch(self, batch: List[Dict]):
        lite_ids = [x['lite_ids'] for x in batch]
        scores_lite = await asyncio.to_thread(self._run_aux_model, self.model_lite, lite_ids, "rerank")

        survivors_1 = []
        for i, s in enumerate(scores_lite):
            if s < 0.02: continue 
            batch[i]['lite_score'] = float(s)
            survivors_1.append(batch[i])
        if not survivors_1: return

        heavy_ids, survivors_2 = [], []
        for item in survivors_1:
            cached = self._check_dedup(item['claim_text'])
            if cached:
                item['analysis'] = cached
                item['cache_hit'] = True
                survivors_2.append(item)
                continue
            if item['lite_score'] > 0.98:
                item['heavy_score'] = 1.0
                survivors_2.append(item)
            else:
                heavy_ids.append(item['heavy_ids'])

        if heavy_ids:
            scores_heavy = await asyncio.to_thread(self._run_aux_model, self.model_heavy, heavy_ids, "rerank")
            heavy_idx = 0
            for item in survivors_1:
                if item.get('cache_hit') or item.get('heavy_score'): continue
                s = float(scores_heavy[heavy_idx])
                heavy_idx += 1
                if s > 0.75:
                    item['heavy_score'] = s
                    survivors_2.append(item)

        if not survivors_2: return

        gen_tasks, gen_indices = [], []
        for i, item in enumerate(survivors_2):
            if item.get('cache_hit'): continue
            system_p = "You are an expert fact-checker. Analyze the Claim against the Evidence. First, output a thinking block <think>...</think> where you analyze the evidence, check for contradictions, and cite specific sentences. Then, output a strictly valid JSON object with the following keys: synthesis, critique, reasoning_trace, entities (list), graph_relations (list), citations (list of ints), verdict, falsity_score (0-9), falsity_explanation, search_query."
            txt = self.tok_main.apply_chat_template([{"role": "system", "content": system_p}, {"role": "user", "content": f"Claim: {item['claim_text']}\nEvidence: {item['doc_text'][:15000]}"}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            req_id = f"{item['claim_id']}-{uuid.uuid4()}"
            gen_tasks.append(self._generate_async(req_id, txt))
            gen_indices.append(i)

        if gen_tasks:
            results = await asyncio.gather(*gen_tasks)
            for idx, raw_text in enumerate(results):
                if not raw_text: continue
                try:
                    json_match = re.search(r'</think>\s*(\{.*\})', raw_text, re.DOTALL)
                    if not json_match: json_match = re.search(r'(\{.*\})$', raw_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                        expanded = {
                            "synthesis": data.get('synthesis', ''), "critique": data.get('critique', ''), "reasoning_trace": data.get('reasoning_trace', ''),
                            "entities": data.get('entities', []), "graph_relations": data.get('graph_relations', []), "citations": data.get('citations', []),
                            "verdict": data.get('verdict', 'Unknown'), "falsity_score": float(data.get('falsity_score', 0)) / 9.0, 
                            "falsity_explanation": data.get('falsity_explanation', ''), "search_query": data.get('search_query', '')
                        }
                        item_idx = gen_indices[idx]
                        survivors_2[item_idx]['analysis'] = expanded
                        self._update_dedup(survivors_2[item_idx]['claim_text'], expanded)
                except Exception: continue

        final_results, embed_texts = [], []
        for item in survivors_2:
            if 'analysis' not in item: continue
            graph_str = ', '.join(str(x) for x in item['analysis']['graph_relations'])
            rich = f"Claim: {item['claim_text']}\nVerdict: {item['analysis']['verdict']}\nScore: {item['analysis']['falsity_score']}\nJustification: {item['analysis']['falsity_explanation']}\nSynthesis: {item['analysis']['synthesis']}\nCritique: {item['analysis']['critique']}\nGraph: {graph_str}\nReasoning: {item['analysis']['reasoning_trace']}"
            embed_texts.append(rich)
            final_results.append(item)
            
        if not final_results: return

        encoded_embed = self.tok_dense(embed_texts, padding=True, truncation=True, max_length=1024)["input_ids"]
        b_vecs = await asyncio.to_thread(self._run_aux_model, self.model_dense, encoded_embed, "dense")
        s_ind, s_val = await asyncio.to_thread(self._run_aux_model, self.model_sparse, None, "sparse", texts=embed_texts)
        
        for i, item in enumerate(final_results):
            item['dense_vec'] = b_vecs[i]
            item['sparse_idx'] = s_ind[i]
            item['sparse_val'] = s_val[i]
            item['record_uuid'] = str(uuid.uuid4())

        # Blocking Save to prevent backlog
        await asyncio.to_thread(self._save_batch, final_results)

# Orchestrator

@app.local_entrypoint()
def main(input_file: str):
    if not input_file: 
        print("❌ Usage: modal run y.py --input-file data.arrow")
        return
    
    print("⚡ Triggering Remote Dataset Loader...")
    processed_items = DatasetLoader().process_and_rank.remote(input_file)
    
    if not processed_items:
        print("❌ No items returned.")
        return
    
    print("⚡ Checking for existing progress in DB...")
    # Use lightweight scanner (CPU)
    existing_ids = set(DBScanner().get_existing_ids.remote())
    print(f"   Found {len(existing_ids)} previously processed claims.")
    
    initial_count = len(processed_items)
    processed_items = [x for x in processed_items if x['claim_id'] not in existing_ids]
    skipped_count = initial_count - len(processed_items)
    
    if skipped_count > 0:
        print(f"⏩ SKIPPING {skipped_count} items (already in DB).")
        print(f"🔥 Dispatching {len(processed_items)} Remaining items to H200...")
    else:
        print(f"🔥 Dispatching {len(processed_items)} items (Fresh Run) to H200...")
    
    if not processed_items:
        print("✅ Job Complete (Nothing new to process).")
        return

    refinery = GodModeRefinery()
    
    async def driver():
        futs = []
        for i in range(0, len(processed_items), pipeline_batch_size):
            batch = processed_items[i:i+pipeline_batch_size]
            futs.append(asyncio.create_task(refinery.process_batch.remote.aio(batch)))
            
            # Allow event loop to breathe
            if len(futs) > 50:
                done, _ = await asyncio.wait(futs, return_when=asyncio.FIRST_COMPLETED)
                futs = [f for f in futs if not f.done()]
        await asyncio.gather(*futs)

    print("🚀 Starting GPU Monolith...")
    asyncio.run(driver())
    print("👋 Job Fully Complete.")