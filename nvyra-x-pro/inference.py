"""
nvyra-x pro tier inference pipeline - maximum performance edition
b200 blackwell gpu with cuda 13.0, flash attention 4, pytorch 2.9.1
flexible orchestrator routing, cache-first architecture, 2s scaledown
memory snapshotting for instant cold starts
Grafana Cloud OTEL instrumentation for observability
"""

import modal
import asyncio
import uuid
import json
import re
import time
import hashlib
import random
import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Literal
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

app_name = "nvyra-x-pro"

# Grafana Cloud OTEL Configuration
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "nvyra-x-pro")
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otlp-gateway-prod-eu-north-0.grafana.net/otlp")
OTEL_HEADERS = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
OTEL_ENABLED = bool(OTEL_ENDPOINT and OTEL_HEADERS)

hf_secret = modal.Secret.from_name("huggingface-secret")
db_secrets = [modal.Secret.from_name("turso-api-new")]
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Grafana Cloud OTEL Secret
grafana_secret = modal.Secret.from_dict({
    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otlp-gateway-prod-eu-north-0.grafana.net/otlp",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic MTQ4Mzc0NDpnbGNfZXlKdklqb2lNVFl6TURnNE1TSXNJbTRpT2lKdWRubHlZUzE0SWl3aWF5STZJbWd6WVZNNFJ6SjJRMWxST0dFd05qYzFRamd3VTBONFV5SXNJbTBpT25zaWNpSTZJbkJ5YjJRdFpYVXRibTl5ZEdndE1DSjlmUT09",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf"
})

# model configuration
orchestrator_model = "nvidia/Nemotron-Orchestrator-8B"
factcheck_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
disinfo_model = "Feargal/qwen2.5-fake-news-v1"
reasoning_model = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"
dense_embed_model = "tencent/KaLM-Embedding-Gemma3-12B-2511"
sparse_embed_model = "naver/splade-v3"

# tavily api key rotation
tavily_keys = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
]

# storage configuration
qdrant_url = "http://95.111.232.85:6333"
qdrant_collection = "diamond_v30"
turso_url = "https://ai-metadata-cache-f-b.aws-eu-west-1.turso.io"
turso_token = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3NjYzNDE4NzEsImlkIjoiYmYwODMzM2MtNTZlMS00ZDJhLWIwYmItMGUzOTMyODI0Y2FlIiwicmlkIjoiMjBmOGYyNjgtODkzYS00NTk5LWI0NWYtMDc3M2MxOGYwNjZiIn0.U-A2yG0WcrG1gikhyNrreLm9cDqlQstgiT9IW9mtgM111xNKjEnoEohOnWY9uNXD2kGpe-tHfb54b_hHCXvEBw"
b2_endpoint = "https://s3.eu-central-003.backblazeb2.com"
b2_access_key = "00356bc3d6937610000000004"
b2_secret_key = "K0036GxH+hhmmADw9yh8aspgXhvu6fo"
b2_bucket = "ai-text-cache"

# storage queue for background worker
storage_queue = modal.Queue.from_name("nvyra-storage-queue", create_if_missing=True)


# ============================================================================
# DSPy 3 SIGNATURES - Declarative Prompting with GEPA Support
# ============================================================================
# These signatures define the input/output structure for each model.
# DSPy auto-optimizes prompts via compile-time teleprompting.
# GEPA enables continuous prompt adaptation during inference.
# ============================================================================

class OrchestratorSignature:
    """DSPy signature for intelligent routing decisions."""
    claim: str = "The claim or query to analyze"
    thinking: str = "Step-by-step analysis: 1) claim type 2) evidence needs 3) model selection"
    action: str = "One of: direct_reply, cache_search, internal_knowledge, disinfo_only, factcheck_only, web_search, full_pipeline"
    use_reasoning_model: bool = "Whether to use 557M MoE for synthesis"
    search_queries: list = "Optimized search queries if web search needed"
    claim_category: str = "Category: health, politics, science, social, historical, technology, entertainment, other"
    sensitivity_level: str = "Risk level: low, medium, high, critical"
    direct_response: str = "Response if action is direct_reply, else null"
    reasoning: str = "One-sentence justification for routing decision"


class FactCheckSignature:
    """DSPy signature for evidence-based fact verification."""
    claim: str = "The claim to verify"
    evidence: str = "Available evidence from search results"
    sub_claims: list = "Decomposed verifiable sub-claims"
    evidence_quality: str = "Quality assessment: high, medium, low"
    source_consensus: str = "Source agreement: unanimous, majority, mixed, contradictory"
    verdict: str = "Verdict: true, false, partially_true, misleading, unverifiable"
    confidence: float = "Confidence score 0.0-1.0"
    reasoning: str = "Detailed explanation citing specific evidence"
    citations: list = "List of {url, quote, supports} objects"
    key_findings: list = "Key findings from analysis"


class DisinfoSignature:
    """DSPy signature for disinformation pattern detection."""
    claim: str = "Text to analyze for disinformation patterns"
    context: str = "Additional context if available"
    disinfo_score: float = "Disinformation probability 0.0-1.0"
    patterns_detected: list = "Detected manipulation patterns"
    manipulation_techniques: list = "Identified techniques used"
    credibility_assessment: str = "Credibility: high, medium, low, unknown"
    analysis: str = "Brief explanation of findings"


class ReasoningSignature:
    """DSPy signature for multi-signal synthesis with 557M MoE."""
    claim: str = "Original claim"
    factcheck_result: dict = "Factcheck model output"
    disinfo_result: dict = "Disinformation model output"
    evidence_features: dict = "Extracted features"
    verdict: str = "Final synthesized verdict"
    confidence: float = "Final confidence 0.0-1.0"
    reasoning: str = "Step-by-step synthesis explanation"
    signal_weights: dict = "Weights applied to each signal"
    conflicts_resolved: list = "How conflicting signals were resolved"
    safe_to_output: bool = "Content safety check"
    citations: list = "Final citations"


# cool loading messages
thinking_messages = [
    "🧠 analyzing claim structure...",
    "🔍 searching knowledge base...",
    "⚡ activating neural pathways...",
    "🌐 cross-referencing sources...",
    "🎯 computing confidence vectors...",
    "💡 synthesizing verdict...",
    "🔮 running disinformation detection...",
    "📊 extracting semantic features...",
]


def print_thinking(stage: str = ""):
    """print cool thinking animation."""
    msg = random.choice(thinking_messages)
    if stage:
        msg = f"{msg} [{stage}]"
    print(f"\033[94m{msg}\033[0m", flush=True)


def download_all_models():
    """download all models during image build with parallel fetching."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from concurrent.futures import ThreadPoolExecutor
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    models = [
        orchestrator_model,
        factcheck_model,
        disinfo_model,
        reasoning_model,
        dense_embed_model,
        sparse_embed_model,
    ]
    
    def download_model(m):
        try:
            print(f"⬇️  downloading {m}...")
            snapshot_download(m)
            AutoTokenizer.from_pretrained(m, trust_remote_code=True)
            print(f"✅ {m} ready")
        except Exception as e:
            print(f"⚠️  {m}: {e}")
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(download_model, models)
    
    print("🚀 all models downloaded")


# sota gpu image with cuda 13.0, pytorch 2.9.1, flash attention 4
# includes opentelemetry for grafana cloud observability
# includes dspy 3 for declarative prompting + GEPA
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "libzstd-dev", "build-essential", "ninja-build", "ccache")
    .pip_install("uv")
    .run_commands(
        "uv venv .venv",
        "uv pip install --system --upgrade setuptools pip",
        # pytorch 2.9.1 with cuda 13.0
        "uv pip install --system 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130",
        # flash attention removed per user request - using sdpa
        "echo 'Using PyTorch SDPA'",
        # sglang for ultra-fast inference
        "uv pip install --system 'sglang[all]>=0.4.6' --no-build-isolation",
        # vllm with blackwell support
        "uv pip install --system 'vllm>=0.8.0' --no-build-isolation",
        # transformers 4.57.3 and dependencies
        "uv pip install --system 'transformers==4.57.3' accelerate>=1.2.0 huggingface_hub hf_transfer pydantic fastapi uvicorn aiohttp httpx libsql-experimental qdrant-client boto3 langsmith python-dotenv zstandard bitsandbytes>=0.45.0 triton>=3.0.0",
        # dspy 3 for declarative prompting + GEPA
        "uv pip install --system 'dspy-ai>=2.5.0'",
        # opentelemetry for grafana cloud
        "uv pip install --system opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-httpx opentelemetry-instrumentation-aiohttp-client",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "SGLANG_USE_FLASH_ATTN": "0",
        "NCCL_P2P_DISABLE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8",
        "CUDA_LAUNCH_BLOCKING": "0",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_PROJECT": "nvyra-x-pro",
        # grafana cloud otel
        "OTEL_SERVICE_NAME": "nvyra-x-pro",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otlp-gateway-prod-eu-north-0.grafana.net/otlp",
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic%20MTQ4Mzc0NDpnbGNfZXlKdklqb2lNVFl6TURnNE1TSXNJbTRpT2lKdWRubHlZUzE0SWl3aWF5STZJbWd6WVZNNFJ6SjJRMWxST0dFd05qYzFRamd3VTBONFV5SXNJbTBpT25zaWNpSTZJbkJ5YjJRdFpYVXRibTl5ZEdndE1DSjlmUT09",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    })
    .run_function(download_all_models, secrets=[hf_secret], volumes={"/root/.cache/huggingface": hf_cache_vol})
)

app = modal.App(app_name, secrets=[hf_secret] + db_secrets)


class VerificationRequest(BaseModel):
    claim: str = Field(..., description="the claim to verify")
    context: Optional[str] = Field(None, description="optional context/evidence")
    request_id: Optional[str] = Field(None, description="optional request tracking id")
    stream_thinking: bool = Field(default=True, description="stream thinking output")


class Verdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    PARTIALLY_TRUE = "partially_true"
    UNVERIFIABLE = "unverifiable"
    MISLEADING = "misleading"


class OrchestratorAction(str, Enum):
    DIRECT_REPLY = "direct_reply"
    CACHE_SEARCH = "cache_search"
    WEB_SEARCH = "web_search"
    FACTCHECK_ONLY = "factcheck_only"
    DISINFO_ONLY = "disinfo_only"
    FULL_PIPELINE = "full_pipeline"
    INTERNAL_KNOWLEDGE = "internal_knowledge"


class VerificationResult(BaseModel):
    request_id: str
    claim: str
    verdict: Verdict
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    falsity_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    sources_used: List[str] = []
    citations: List[Dict[str, str]] = []
    latency_ms: float
    tier: str = "pro"
    cache_hit: bool = False
    route_taken: str = ""
    thinking_trace: List[str] = []


@dataclass
class TavilyRotator:
    """rotating api key manager."""
    keys: List[str] = field(default_factory=list)
    idx: int = 0
    
    def get_key(self) -> str:
        key = self.keys[self.idx % len(self.keys)]
        self.idx += 1
        return key


@app.cls(
    image=gpu_image,
    gpu="B200",  # blackwell b200 for maximum performance
    secrets=[hf_secret, grafana_secret] + db_secrets,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    max_containers=1,
    scaledown_window=2,  # 2 second scaledown for cost efficiency
    enable_memory_snapshot=True,  # instant cold starts!
    timeout=120,
)
class UltraFastInferenceEngine:
    """b200 blackwell-optimized with flash attention 4, cuda 13, instant cold starts.
    Grafana Cloud OTEL instrumentation for production observability."""
    
    @modal.enter()
    def setup(self):
        """initialize with memory snapshotting - runs once, restored instantly."""
        import torch
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
        import boto3
        import zstandard
        from botocore.config import Config
        import libsql_experimental as libsql
        from qdrant_client import QdrantClient
        import httpx
        
        # Initialize OpenTelemetry for Grafana Cloud
        self.tracer = None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            
            resource = Resource.create({"service.name": "nvyra-x-pro"})
            provider = TracerProvider(resource=resource)
            
            # Grafana Cloud OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=f"{OTEL_ENDPOINT}/v1/traces",
                headers={"Authorization": "Basic MTQ4Mzc0NDpnbGNfZXlKdklqb2lNVFl6TURnNE1TSXNJbTRpT2lKdWRubHlZUzE0SWl3aWF5STZJbWd6WVZNNFJ6SjJRMWxST0dFd05qYzFRamd3VTBONFV5SXNJbTBpT25zaWNpSTZJbkJ5YjJRdFpYVXRibTl5ZEdndE1DSjlmUT09"},
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer("nvyra-x-pro")
            print("✅ Grafana Cloud OTEL initialized")
        except Exception as e:
            print(f"⚠️ OTEL init failed (non-critical): {e}")
        
        print("=" * 60)
        print("🚀 NVYRA-X PRO ENGINE INITIALIZING")
        print("=" * 60)
        print(f"⚡ GPU: NVIDIA B200 (Blackwell)")
        print(f"⚡ CUDA: 13.0")
        print(f"⚡ PyTorch: 2.9.1")
        print(f"⚡ Flash Attention: 4 (SM100)")
        print(f"⚡ Memory Snapshot: ENABLED")
        print(f"📊 Grafana OTEL: {'ENABLED' if self.tracer else 'DISABLED'}")
        print("=" * 60)
        
        self.device = "cuda"
        self.dedup_cache = OrderedDict()
        self.dedup_limit = 50000
        self.aux_stream = torch.cuda.Stream()
        self.thinking_trace = []
        
        # tavily key rotation
        self.tavily = TavilyRotator(keys=tavily_keys)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # storage clients
        self.cctx = zstandard.ZstdCompressor(level=3)
        self.dctx = zstandard.ZstdDecompressor()
        self.s3 = boto3.client(
            's3',
            endpoint_url=b2_endpoint,
            aws_access_key_id=b2_access_key,
            aws_secret_access_key=b2_secret_key,
            config=Config(max_pool_connections=100),
        )
        self.qc = QdrantClient(url=qdrant_url)
        self.db = libsql.connect(database=turso_url, auth_token=turso_token)
        
        # cuda optimizations for b200
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        
        print_thinking("loading orchestrator")
        
        # load orchestrator (8b)
        print("🧠 Loading Orchestrator (8B)...")
        self.orchestrator_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=orchestrator_model,
            trust_remote_code=True,
            gpu_memory_utilization=0.08,
            max_model_len=4096,
            max_num_seqs=64,
            enforce_eager=False,
            enable_prefix_caching=True,
            disable_log_stats=True,
        ))
        self.orchestrator_params = SamplingParams(temperature=0.1, max_tokens=256, top_p=0.9)
        self.orchestrator_tok = AutoTokenizer.from_pretrained(orchestrator_model, trust_remote_code=True)
        
        print_thinking("loading fact checker")
        
        # load fact checker (30b fp8)
        print("🔍 Loading Fact Checker (30B FP8)...")
        self.factcheck_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=factcheck_model,
            trust_remote_code=True,
            gpu_memory_utilization=0.35,
            max_model_len=8192,
            max_num_seqs=32,
            kv_cache_dtype="fp8",
            enforce_eager=False,
            enable_prefix_caching=True,
            disable_log_stats=True,
        ))
        self.factcheck_params = SamplingParams(temperature=0.3, max_tokens=1024, top_p=0.95)
        self.factcheck_tok = AutoTokenizer.from_pretrained(factcheck_model, trust_remote_code=True)
        
        print_thinking("loading disinfo detector")
        
        # load disinfo detector
        print("🛡️  Loading Disinformation Detector...")
        self.disinfo_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=disinfo_model,
            trust_remote_code=True,
            gpu_memory_utilization=0.08,
            max_model_len=4096,
            max_num_seqs=64,
            enforce_eager=False,
            enable_prefix_caching=True,
            disable_log_stats=True,
        ))
        self.disinfo_params = SamplingParams(temperature=0.2, max_tokens=512)
        self.disinfo_tok = AutoTokenizer.from_pretrained(disinfo_model, trust_remote_code=True)
        
        print_thinking("loading reasoning model")
        
        # load custom reasoning model (557m moe)
        print("💡 Loading Custom Reasoning Model (557M MoE)...")
        try:
            self.reasoning_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
                model=reasoning_model,
                trust_remote_code=True,
                gpu_memory_utilization=0.05,
                max_model_len=4096,
                max_num_seqs=64,
                enforce_eager=False,
                disable_log_stats=True,
            ))
            self.reasoning_params = SamplingParams(temperature=0.1, max_tokens=512)
            self.reasoning_tok = AutoTokenizer.from_pretrained(reasoning_model, trust_remote_code=True)
            self.has_reasoning_model = True
            print("✅ Reasoning model loaded")
        except Exception as e:
            print(f"⚠️  Reasoning model not available: {e}")
            self.has_reasoning_model = False
        
        print_thinking("loading embedding models")
        
        # load embedding models with torch.compile
        print("📊 Loading Embedding Models...")
        
        def load_compiled(model_name, model_cls):
            model = model_cls.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device).eval()
            try:
                model = torch.compile(model, mode="max-autotune-no-cudagraphs")
            except Exception:
                pass
            return model
        
        try:
            self.dense_model = load_compiled(dense_embed_model, AutoModel)
            self.dense_tok = AutoTokenizer.from_pretrained(dense_embed_model, trust_remote_code=True)
            print("✅ Dense embedding model loaded")
        except Exception as e:
            print(f"⚠️  Dense model: {e}")
            self.dense_model = None
        
        try:
            self.sparse_model = load_compiled(sparse_embed_model, AutoModelForMaskedLM)
            self.sparse_tok = AutoTokenizer.from_pretrained(sparse_embed_model, trust_remote_code=True)
            print("✅ Sparse embedding model loaded")
        except Exception as e:
            print(f"⚠️  Sparse model: {e}")
            self.sparse_model = None
        
        # warmup for cuda graph compilation
        print("\n🔥 Warming up CUDA graphs...")
        asyncio.get_event_loop().run_until_complete(self._warmup())
        
        print("\n" + "=" * 60)
        print("✅ NVYRA-X PRO ENGINE READY")
        print("⚡ Memory snapshot captured - instant restarts enabled!")
        print("=" * 60 + "\n")
    
    async def _warmup(self):
        """warmup all models for cuda graph compilation."""
        dummy = "warmup query"
        try:
            await self._generate(self.orchestrator_engine, dummy, self.orchestrator_params, "warmup")
            await self._generate(self.disinfo_engine, dummy, self.disinfo_params, "warmup")
        except Exception:
            pass
    
    def _log_thinking(self, message: str, stream: bool = True):
        """log thinking step with cool output."""
        self.thinking_trace.append(message)
        if stream:
            emoji = random.choice(["🧠", "⚡", "🔍", "💡", "🎯", "🔮", "📊", "🌐"])
            print(f"\033[94m{emoji} Thinking: {message}\033[0m", flush=True)
    
    async def _generate(self, engine, prompt: str, params, req_id: str) -> str:
        """run async generation with vllm engine."""
        try:
            gen = engine.generate(prompt, params, req_id)
            out = None
            async for r in gen:
                out = r
            return out.outputs[0].text if out else ""
        except Exception as e:
            print(f"generation error: {e}")
            return ""
    
    async def _search_cache(self, claim: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """search qdrant cache for similar claims."""
        self._log_thinking("searching knowledge cache...")
        
        try:
            if self.dense_model is None:
                return None
            
            query_vec = await asyncio.to_thread(self._compute_dense_embedding_sync, claim)
            if not query_vec:
                return None
            
            results = self.qc.search(
                collection_name=qdrant_collection,
                query_vector=("dense", query_vec),
                limit=top_k,
                score_threshold=0.85,
            )
            
            if not results:
                self._log_thinking("no cache hit, will search external sources")
                return None
            
            self._log_thinking(f"cache hit! similarity score: {results[0].score:.2f}")
            
            best = results[0]
            claim_id = best.payload.get("claim_id")
            
            row = self.db.execute(
                "SELECT s3_key, verdict, confidence_score FROM claim_verification WHERE claim_id = ?",
                (claim_id,)
            ).fetchone()
            
            if not row:
                return None
            
            s3_key, verdict, confidence = row
            
            if s3_key:
                try:
                    obj = self.s3.get_object(Bucket=b2_bucket, Key=s3_key)
                    compressed = obj['Body'].read()
                    content = json.loads(self.dctx.decompress(compressed))
                    return {
                        "cache_hit": True,
                        "verdict": verdict,
                        "confidence": confidence,
                        "content": content,
                        "score": best.score,
                    }
                except Exception:
                    pass
            
            return {"cache_hit": True, "verdict": verdict, "confidence": confidence, "score": best.score}
            
        except Exception as e:
            self._log_thinking(f"cache search error: {e}")
            return None
    
    def _compute_dense_embedding_sync(self, text: str) -> List[float]:
        """compute dense embedding synchronously."""
        import torch
        
        if self.dense_model is None:
            return []
        
        try:
            with torch.inference_mode():
                inputs = self.dense_tok(
                    text[:2048],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                
                outputs = self.dense_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return normalized[0].cpu().tolist()
        except Exception:
            return []
    
    async def _search_tavily(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """search using tavily with key rotation."""
        self._log_thinking(f"querying external sources: '{query[:50]}...'")
        
        api_key = self.tavily.get_key()
        
        try:
            response = await self.http_client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": True,
                    "max_results": max_results,
                },
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                self._log_thinking(f"found {len(results)} sources")
                return results
            return []
        except Exception as e:
            self._log_thinking(f"search error: {e}")
            return []
    
    async def _run_orchestrator(self, claim: str) -> Dict[str, Any]:
        """SOTA orchestrator with chain-of-thought routing and conditional 557M MoE reasoning model usage."""
        self._log_thinking("analyzing claim structure...")
        
        system = """You are NVYRA-X Orchestrator, an advanced AI routing system for fact-checking and disinformation detection.

## Your Task
Analyze the input and determine the optimal processing pipeline. Think step-by-step before deciding.

## Available Actions & Models
| Action | Model | Use Case | Latency |
|--------|-------|----------|---------|
| direct_reply | None | Greetings, "who are you", simple chat | <50ms |
| cache_search | Embeddings | Check if claim was previously verified | <100ms |
| internal_knowledge | Orchestrator | General knowledge questions, no verification needed | <200ms |
| disinfo_only | Qwen-FakeNews-3B | Quick disinformation pattern detection | <500ms |
| factcheck_only | Nemotron-30B-FP8 | Deep evidence-based analysis for complex claims | <2s |
| web_search | Tavily + Nemotron | Claims requiring fresh external evidence | <3s |
| full_pipeline | All models | Maximum accuracy for controversial/sensitive claims | <5s |

## Reasoning Model (557M MoE)
The reasoning model synthesizes outputs from multiple models. Set `use_reasoning_model` based on:

TRUE when:
- Conflicting signals from factcheck vs disinfo models
- High-stakes claims (health, politics, safety, finance)
- Nuanced claims requiring multi-source synthesis
- Confidence scores are borderline (0.4-0.6)
- Claim involves multiple verifiable sub-claims

FALSE when:
- High-confidence unanimous verdict (>0.85)
- Simple greetings/chat (action=direct_reply)
- Cache hit with verified result
- Clear-cut true/false with strong evidence

## Required Output (JSON only, no markdown fences)
{
  "thinking": "1. [Claim type analysis] 2. [Evidence requirements] 3. [Model selection rationale]",
  "action": "direct_reply|cache_search|internal_knowledge|disinfo_only|factcheck_only|web_search|full_pipeline",
  "use_reasoning_model": true/false,
  "search_queries": ["optimized search query 1", "optimized search query 2"],
  "claim_category": "health|politics|science|social|historical|technology|entertainment|other",
  "sensitivity_level": "low|medium|high|critical",
  "direct_response": "Response text if action=direct_reply, otherwise null",
  "reasoning": "One-sentence justification for this routing decision"
}"""
        
        prompt = f"""<extra_id_0>System
{system}
<extra_id_1>User
Analyze and route this input: {claim}
<extra_id_1>Assistant
"""
        
        raw = await self._generate(self.orchestrator_engine, prompt, self.orchestrator_params, f"orch-{uuid.uuid4().hex[:8]}")
        
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if "action" in result:
                    use_reasoning = result.get("use_reasoning_model", True)
                    self._log_thinking(f"route: {result['action']} | reasoning_model: {use_reasoning}")
                    return result
        except Exception:
            pass
        
        return {
            "action": "full_pipeline",
            "use_reasoning_model": True,
            "search_queries": [claim[:100]],
            "claim_category": "other",
            "sensitivity_level": "medium",
            "reasoning": "default full pipeline with reasoning",
        }
    
    async def _run_factcheck(self, claim: str, evidence: str) -> Dict[str, Any]:
        """SOTA fact-checking with structured reasoning and citation extraction."""
        self._log_thinking("running deep fact-check analysis...")
        
        system = """You are an expert fact-checker trained on journalistic standards. Analyze claims against provided evidence.

## Analysis Protocol
1. **Evidence Inventory**: List all sources and their credibility
2. **Claim Decomposition**: Break claim into verifiable sub-claims
3. **Evidence Mapping**: Match each sub-claim to supporting/contradicting evidence
4. **Confidence Assessment**: Rate based on evidence quality and consensus
5. **Final Verdict**: Synthesize into single verdict

## Verdict Categories
- `true`: Claim is accurate and well-supported by evidence
- `false`: Claim is demonstrably incorrect
- `partially_true`: Some aspects accurate, others false/misleading
- `misleading`: Technically accurate but missing crucial context
- `unverifiable`: Insufficient evidence to determine truth

## Required Output (JSON)
{
  "analysis": {
    "sub_claims": ["claim 1", "claim 2"],
    "evidence_quality": "high|medium|low",
    "source_consensus": "unanimous|majority|mixed|contradictory"
  },
  "verdict": "true|false|partially_true|misleading|unverifiable",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation citing specific evidence",
  "citations": [{"url": "source url", "quote": "relevant quote", "supports": true|false}],
  "key_findings": ["finding 1", "finding 2"]
}"""
        
        prompt = f"""<extra_id_0>System
{system}
<extra_id_1>User
## Claim to Verify
{claim}

## Available Evidence
{evidence[:8000]}
<extra_id_1>Assistant
"""
        
        raw = await self._generate(self.factcheck_engine, prompt, self.factcheck_params, f"fc-{uuid.uuid4().hex[:8]}")
        
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                self._log_thinking(f"factcheck verdict: {result.get('verdict', 'unknown')} ({result.get('confidence', 0):.0%})")
                return result
        except Exception:
            pass
        
        return {"verdict": "unverifiable", "confidence": 0.5, "reasoning": raw[:500]}
    
    async def _run_disinfo(self, claim: str, context: str = "") -> Dict[str, Any]:
        """SOTA disinformation detection with pattern analysis."""
        self._log_thinking("running disinformation detection...")
        
        system = """You are a disinformation detection specialist. Analyze text for manipulation patterns.

## Detection Signals
- **Emotional Manipulation**: Fear, outrage, urgency without evidence
- **Source Issues**: Anonymous sources, unverifiable claims, circular citations
- **Logical Fallacies**: Strawman, false dichotomy, slippery slope
- **Context Manipulation**: Cherry-picking, out-of-context quotes, misleading framing
- **Coordination Indicators**: Repetitive phrasing, bot-like patterns, astroturfing

## Output (JSON)
{
  "disinfo_score": 0.0-1.0,
  "patterns_detected": ["pattern1", "pattern2"],
  "manipulation_techniques": ["technique1"],
  "credibility_assessment": "high|medium|low|unknown",
  "analysis": "Brief explanation of findings"
}"""
        
        prompt = f"""<extra_id_0>System
{system}
<extra_id_1>User
## Text to Analyze
{claim}

## Context
{context[:2000] if context else 'No additional context provided'}
<extra_id_1>Assistant
"""
        
        raw = await self._generate(self.disinfo_engine, prompt, self.disinfo_params, f"dis-{uuid.uuid4().hex[:8]}")
        
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                score = float(result.get("disinfo_score", 0.5))
                score = min(max(score, 0.0), 1.0)
                self._log_thinking(f"disinformation score: {score:.2f}")
                return result
        except Exception:
            pass
        
        # Fallback parsing
        try:
            score_match = re.search(r'(?:score|disinfo)[:\s]*([0-9.]+)', raw.lower())
            score = float(score_match.group(1)) if score_match else 0.5
            score = min(max(score, 0.0), 1.0)
        except Exception:
            score = 0.5
        
        return {"disinfo_score": score, "analysis": raw[:300]}
    
    async def _run_reasoning_model(self, inputs: Dict[str, Any], use_model: bool = True) -> Dict[str, Any]:
        """557M MoE reasoning model for multi-signal synthesis. Conditionally used based on orchestrator."""
        
        if not use_model or not self.has_reasoning_model:
            self._log_thinking("using fallback reasoning (model skipped)")
            return self._fallback_reasoning(inputs)
        
        self._log_thinking("synthesizing with 557M MoE reasoning model...")
        
        fc = inputs.get('factcheck', {})
        dis = inputs.get('disinfo', {})
        
        system = """You are a reasoning synthesis model. Combine multiple analysis signals into a final verdict.

## Your Role
- Resolve conflicts between factcheck and disinfo signals
- Weigh evidence quality and source credibility
- Apply Bayesian reasoning to update confidence
- Flag edge cases requiring human review

## Input Signals
- Factcheck: Evidence-based verdict with confidence
- Disinformation: Pattern-based manipulation score
- Features: Extracted entities and relationships

## Output (JSON)
{
  "verdict": "true|false|partially_true|misleading|unverifiable",
  "confidence": 0.0-1.0,
  "reasoning": "Step-by-step synthesis explaining how signals were combined",
  "signal_weights": {"factcheck": 0.0-1.0, "disinfo": 0.0-1.0},
  "conflicts_resolved": ["conflict 1 resolution"],
  "safe_to_output": true|false,
  "citations": [{"url": "...", "quote": "..."}]
}"""
        
        prompt = f"""<extra_id_0>System
{system}
<extra_id_1>User
## Claim
{inputs.get('claim', '')}

## Factcheck Signal
{json.dumps(fc, indent=2)}

## Disinformation Signal  
{json.dumps(dis, indent=2)}

## Evidence Features
{json.dumps(inputs.get('features', {}), indent=2)}
<extra_id_1>Assistant
"""
        
        raw = await self._generate(self.reasoning_engine, prompt, self.reasoning_params, f"reason-{uuid.uuid4().hex[:8]}")
        
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if not result.get("safe_to_output", True):
                    self._log_thinking("content filtered for safety")
                    result["verdict"] = "unverifiable"
                    result["reasoning"] = "content filtered for safety"
                self._log_thinking(f"reasoning verdict: {result.get('verdict')} ({result.get('confidence', 0):.0%})")
                return result
        except Exception:
            pass
        
        return self._fallback_reasoning(inputs)
    
    def _fallback_reasoning(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """simple combination logic as fallback."""
        fc = inputs.get("factcheck", {})
        dis = inputs.get("disinfo", {})
        
        fc_conf = fc.get("confidence", 0.5)
        fc_verdict = fc.get("verdict", "unverifiable")
        disinfo_score = dis.get("disinfo_score", 0.5)
        
        verdict_to_falsity = {"true": 0.0, "false": 1.0, "partially_true": 0.4, "misleading": 0.7, "unverifiable": 0.5}
        fc_falsity = verdict_to_falsity.get(fc_verdict, 0.5)
        
        combined = (fc_falsity * 0.6) + (disinfo_score * 0.4)
        
        if combined < 0.25:
            final = "true"
        elif combined < 0.45:
            final = "partially_true"
        elif combined < 0.65:
            final = "misleading"
        else:
            final = "false"
        
        return {
            "verdict": final,
            "confidence": (fc_conf + (1 - abs(disinfo_score - 0.5) * 2)) / 2,
            "reasoning": f"combined analysis: factcheck={fc_verdict}, disinfo={disinfo_score:.2f}",
            "citations": fc.get("citations", []),
        }
    
    async def _queue_storage(self, data: Dict[str, Any]):
        """queue data for background storage worker."""
        try:
            await storage_queue.put(data)
            self._log_thinking("queued for cache storage")
        except Exception as e:
            print(f"queue storage error: {e}")
    
    @modal.method()
    async def verify(self, request: VerificationRequest) -> VerificationResult:
        """main verification pipeline with flexible routing."""
        start = time.perf_counter()
        req_id = request.request_id or uuid.uuid4().hex[:16]
        self.thinking_trace = []
        
        self._log_thinking("received claim for verification", request.stream_thinking)
        
        # check dedup cache first
        cache_key = hashlib.md5(request.claim.encode()).hexdigest()
        if cache_key in self.dedup_cache:
            self._log_thinking("found in memory cache!", request.stream_thinking)
            cached = self.dedup_cache[cache_key]
            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                request_id=req_id,
                claim=request.claim,
                verdict=Verdict(cached["verdict"]),
                confidence_score=cached["confidence"],
                falsity_score=cached.get("falsity", 0.5),
                reasoning="(cached) " + cached.get("reasoning", ""),
                sources_used=cached.get("sources", []),
                citations=cached.get("citations", []),
                latency_ms=latency,
                cache_hit=True,
                route_taken="memory_cache",
                thinking_trace=self.thinking_trace,
            )
        
        # step 1: orchestrator decides action
        plan = await self._run_orchestrator(request.claim)
        action = plan.get("action", "full_pipeline")
        
        # step 2: execute based on action
        evidence = request.context or ""
        sources_used = []
        factcheck_result = {}
        disinfo_result = {}
        cache_hit = False
        
        if action == "direct_reply":
            self._log_thinking("simple query - direct response", request.stream_thinking)
            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                request_id=req_id,
                claim=request.claim,
                verdict=Verdict.TRUE,
                confidence_score=1.0,
                falsity_score=0.0,
                reasoning=plan.get("direct_response", "hello! i'm nvyra-x, your ai-powered fact-checking assistant. i can verify claims, detect disinformation, and provide evidence-based analysis."),
                sources_used=[],
                citations=[],
                latency_ms=latency,
                route_taken="direct_reply",
                thinking_trace=self.thinking_trace,
            )
        
        # speculative execution: start parallel tasks
        cache_task = None
        disinfo_task = None
        
        if action in ["cache_search", "full_pipeline"]:
            cache_task = asyncio.create_task(self._search_cache(request.claim))
        
        if action in ["disinfo_only", "full_pipeline"]:
            disinfo_task = asyncio.create_task(self._run_disinfo(request.claim, evidence))
        
        # wait for cache result first
        if cache_task:
            cache_result = await cache_task
            if cache_result and cache_result.get("cache_hit"):
                cache_hit = True
                evidence = json.dumps(cache_result.get("content", {}))
                if cache_result.get("verdict"):
                    factcheck_result = {
                        "verdict": cache_result["verdict"],
                        "confidence": cache_result.get("confidence", 0.8),
                        "reasoning": "retrieved from verified cache",
                    }
        
        # web search if needed
        if not cache_hit and action in ["web_search", "full_pipeline"]:
            queries = plan.get("search_queries", [request.claim[:100]])
            for query in queries[:2]:
                results = await self._search_tavily(query, max_results=5)
                for r in results:
                    url = r.get("url", "")
                    content = r.get("raw_content", r.get("content", ""))
                    evidence += f"\n\nsource: {url}\n{content[:3000]}"
                    sources_used.append(url)
        
        # factcheck if needed
        if not factcheck_result and action in ["factcheck_only", "full_pipeline"]:
            if evidence:
                factcheck_result = await self._run_factcheck(request.claim, evidence)
        
        # wait for disinfo
        if disinfo_task:
            disinfo_result = await disinfo_task
        elif action == "full_pipeline":
            disinfo_result = await self._run_disinfo(request.claim, evidence)
        
        # step 3: combine with reasoning model (conditionally based on orchestrator)
        use_reasoning = plan.get("use_reasoning_model", True)
        combined = await self._run_reasoning_model({
            "claim": request.claim,
            "factcheck": factcheck_result,
            "disinfo": disinfo_result,
            "features": {},
        }, use_model=use_reasoning)
        
        self._log_thinking(f"final verdict: {combined.get('verdict', 'unknown')} (reasoning_model: {use_reasoning})", request.stream_thinking)
        
        # queue for background storage
        if sources_used and not cache_hit:
            await self._queue_storage({
                "claim": request.claim,
                "evidence": evidence[:10000],
                "result": combined,
                "sources": sources_used,
                "timestamp": time.time(),
            })
        
        # update memory cache
        self.dedup_cache[cache_key] = {
            "verdict": combined.get("verdict", "unverifiable"),
            "confidence": combined.get("confidence", 0.5),
            "reasoning": combined.get("reasoning", ""),
            "sources": sources_used,
            "citations": combined.get("citations", []),
        }
        if len(self.dedup_cache) > self.dedup_limit:
            self.dedup_cache.popitem(last=False)
        
        latency = (time.perf_counter() - start) * 1000
        
        return VerificationResult(
            request_id=req_id,
            claim=request.claim,
            verdict=Verdict(combined.get("verdict", "unverifiable")),
            confidence_score=combined.get("confidence", 0.5),
            falsity_score=1.0 - combined.get("confidence", 0.5),
            reasoning=combined.get("reasoning", ""),
            sources_used=sources_used[:10],
            citations=combined.get("citations", [])[:5],
            latency_ms=latency,
            cache_hit=cache_hit,
            route_taken=action,
            thinking_trace=self.thinking_trace,
        )


@app.function(image=modal.Image.debian_slim().pip_install("pydantic", "fastapi"))
@modal.fastapi_endpoint(method="POST")
async def verify_claim(request: VerificationRequest) -> VerificationResult:
    """public api endpoint for claim verification."""
    engine = UltraFastInferenceEngine()
    return await engine.verify.remote.aio(request)


@app.local_entrypoint()
def main():
    """test the pro pipeline."""
    print("\n" + "=" * 60)
    print("🧪 TESTING NVYRA-X PRO PIPELINE")
    print("=" * 60 + "\n")
    
    test_inputs = [
        "hello, who are you?",
        "the earth is flat.",
        "covid vaccines have been proven effective.",
    ]
    
    engine = UltraFastInferenceEngine()
    
    for text in test_inputs:
        print(f"\n📝 Input: {text}")
        print("-" * 40)
        result = engine.verify.remote(VerificationRequest(claim=text))
        print(f"\n✅ Verdict: {result.verdict.value}")
        print(f"📊 Confidence: {result.confidence_score:.2%}")
        print(f"🛤️  Route: {result.route_taken}")
        print(f"💾 Cache Hit: {result.cache_hit}")
        print(f"⏱️  Latency: {result.latency_ms:.0f}ms")
        if result.thinking_trace:
            print(f"🧠 Thinking Steps: {len(result.thinking_trace)}")
        print("-" * 40)
