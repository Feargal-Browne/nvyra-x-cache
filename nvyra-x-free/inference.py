"""
nvyra-x free tier inference pipeline - optimized for speed
cpu-only with qwen3-1.7b-instruct gguf, parallel execution
linear pipeline as per pipeline.md specification
Grafana Cloud OTEL instrumentation for observability
"""

import modal
import asyncio
import uuid
import json
import re
import time
import subprocess
import hashlib
import random
import os
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from collections import deque
import html

app_name = "nvyra-x-free"

# Grafana Cloud OTEL Configuration
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "nvyra-x-free")
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otlp-gateway-prod-eu-north-0.grafana.net/otlp")
OTEL_HEADERS = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")

hf_secret = modal.Secret.from_name("huggingface-secret")
hf_cache_vol = modal.Volume.from_name("huggingface-cache-free", create_if_missing=True)

# Grafana Cloud OTEL Secret
grafana_secret = modal.Secret.from_dict({
    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otlp-gateway-prod-eu-north-0.grafana.net/otlp",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic MTQ4Mzc0NDpnbGNfZXlKdklqb2lNVFl6TURnNE1TSXNJbTRpT2lKdWRubHlZUzE0SWl3aWF5STZJbWd6WVZNNFJ6SjJRMWxST0dFd05qYzFRamd3VTBONFV5SXNJbTBpT25zaWNpSTZJbkJ5YjJRdFpYVXRibTl5ZEdndE1DSjlmUT09",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf"
})

# model configuration - gguf for speed
qwen3_model = "Qwen/Qwen3-4B-Instruct-2507"
qwen3_gguf_file = "qwen3-4b-instruct-q4_k_m.gguf"
qwen_disinfo_model = "Feargal/qwen2.5-fake-news-GGUF"
qwen_disinfo_file = "qwen2.5-1.5b-fake-news-q4_k_m.gguf"
bge_model = "BAAI/bge-small-en-v1.5"
reasoning_model = "Feargal/nvyra-x-reasoning-GGUF"
reasoning_file = "nvyra-x-reasoning-q4_k_m.gguf"

# storage configuration (shared with pro tier)
qdrant_url = "http://95.111.232.85:6333"
qdrant_collection = "diamond_v30"
turso_url = "https://ai-metadata-cache-f-b.aws-eu-west-1.turso.io"
turso_token = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3NjYzNDE4NzEsImlkIjoiYmYwODMzM2MtNTZlMS00ZDJhLWIwYmItMGUzOTMyODI0Y2FlIiwicmlkIjoiMjBmOGYyNjgtODkzYS00NTk5LWI0NWYtMDc3M2MxOGYwNjZiIn0.U-A2yG0WcrG1gikhyNrreLm9cDqlQstgiT9IW9mtgM111xNKjEnoEohOnWY9uNXD2kGpe-tHfb54b_hHCXvEBw"
b2_endpoint = "https://s3.eu-central-003.backblazeb2.com"
b2_access_key = "00356bc3d6937610000000004"
b2_secret_key = "K0036GxH+hhmmADw9yh8aspgXhvu6fo"
b2_bucket = "ai-text-cache"

# search engine rate limits
search_rate_limits = {
    "duckduckgo": 30,
    "bing": 20,
    "brave": 15,
}


# ============================================================================
# DSPy 3 SIGNATURES - Declarative Prompting for CPU Inference
# ============================================================================
# Lightweight signatures for GGUF-based inference with GEPA support.
# ============================================================================

class ClaimExtractorSignature:
    """DSPy signature for claim analysis and routing."""
    claim: str = "The input text to analyze"
    needs_search: bool = "Whether external search is needed"
    search_query: str = "Optimized search query if needed"
    claim_type: str = "Type: factual, opinion, greeting"
    is_simple: bool = "Whether this is a simple query"


class FactCheckSignature:
    """DSPy signature for fact verification."""
    claim: str = "The claim to verify"
    evidence: str = "Available evidence"
    verdict: str = "Verdict: true, false, partially_true, misleading, unverifiable"
    confidence: float = "Confidence 0.0-1.0"
    reasoning: str = "Explanation citing evidence"


class DisinfoSignature:
    """DSPy signature for disinformation detection."""
    text: str = "Text to analyze"
    context: str = "Additional context"
    disinfo_score: float = "Disinformation probability 0.0-1.0"
    analysis: str = "Brief explanation"


def download_models():
    """download gguf models and build llama.cpp during image build."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from sentence_transformers import SentenceTransformer
    import os
    
    os.makedirs("/models", exist_ok=True)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # download qwen3 gguf
    print("downloading qwen3-1.7b-instruct gguf...")
    try:
        hf_hub_download(repo_id=qwen3_model, filename=qwen3_gguf_file, local_dir="/models")
    except Exception as e:
        print(f"qwen3 download: {e}")
        snapshot_download(qwen3_model, local_dir="/models/qwen3")
    
    # download disinfo model gguf
    print("downloading qwen2.5 disinfo gguf...")
    try:
        hf_hub_download(repo_id=qwen_disinfo_model, filename=qwen_disinfo_file, local_dir="/models")
    except Exception as e:
        print(f"disinfo download: {e}")
    
    # download reasoning model gguf
    print("downloading reasoning model gguf...")
    try:
        hf_hub_download(repo_id=reasoning_model, filename=reasoning_file, local_dir="/models")
    except Exception as e:
        print(f"reasoning download: {e}")
    
    # download bge-small for embeddings
    print("downloading bge-small embedding model...")
    try:
        SentenceTransformer(bge_model)
    except Exception as e:
        print(f"bge download: {e}")
    
    print("all models downloaded")


cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential", "cmake", "git", "wget", "curl", "vim",
        "libopenblas-dev", "pkg-config", "libssl-dev", "libcurl4-openssl-dev",
    )
    .run_commands(
        # build llama.cpp with optimizations
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /llama.cpp",
        "cd /llama.cpp && mkdir build && cd build && cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DCMAKE_BUILD_TYPE=Release -DLLAMA_NATIVE=ON && cmake --build . --config Release -j$(nproc)",
        "cp /llama.cpp/build/bin/llama-cli /usr/local/bin/",
    )
    .pip_install(
        "transformers>=4.48.0",
        "torch>=2.0.0",
        "huggingface_hub",
        "hf_transfer",
        "pydantic",
        "fastapi",
        "uvicorn",
        "aiohttp",
        "httpx",
        "beautifulsoup4",
        "lxml",
        "trafilatura",
        "sentence-transformers",
        "qdrant-client",
        "libsql-experimental",
        "boto3",
        "zstandard",
        "langsmith",
        # dspy 3 for declarative prompting + GEPA
        "dspy-ai>=2.5.0",
        # opentelemetry for grafana cloud
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
        "opentelemetry-instrumentation-fastapi",
        "opentelemetry-instrumentation-httpx",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_PROJECT": "nvyra-x-free",
        # grafana cloud otel
        "OTEL_SERVICE_NAME": "nvyra-x-free",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otlp-gateway-prod-eu-north-0.grafana.net/otlp",
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic%20MTQ4Mzc0NDpnbGNfZXlKdklqb2lNVFl6TURnNE1TSXNJbTRpT2lKdWRubHlZUzE0SWl3aWF5STZJbWd6WVZNNFJ6SjJRMWxST0dFd05qYzFRamd3VTBONFV5SXNJbTBpT25zaWNpSTZJbkJ5YjJRdFpYVXRibTl5ZEdndE1DSjlmUT09",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    })
    .run_function(download_models, secrets=[hf_secret], volumes={"/root/.cache/huggingface": hf_cache_vol})
)

app = modal.App(app_name, secrets=[hf_secret])


class VerificationRequest(BaseModel):
    claim: str = Field(..., description="the claim to verify")
    context: Optional[str] = Field(None, description="optional context/evidence")
    request_id: Optional[str] = Field(None, description="optional request tracking id")


class Verdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    PARTIALLY_TRUE = "partially_true"
    UNVERIFIABLE = "unverifiable"
    MISLEADING = "misleading"


class VerificationResult(BaseModel):
    request_id: str
    claim: str
    verdict: Verdict
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    falsity_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    sources_used: List[str] = []
    latency_ms: float
    tier: str = "free"
    cache_hit: bool = False


@dataclass
class RobotsCache:
    """robots.txt parser cache."""
    cache: Dict[str, Tuple[RobotFileParser, float]] = field(default_factory=dict)
    ttl: float = 3600
    
    def get(self, domain: str) -> Optional[RobotFileParser]:
        if domain in self.cache:
            parser, ts = self.cache[domain]
            if time.time() - ts < self.ttl:
                return parser
        return None
    
    def set(self, domain: str, parser: RobotFileParser):
        self.cache[domain] = (parser, time.time())


@app.cls(
    image=cpu_image,
    cpu=8,
    memory=32768,
    secrets=[hf_secret, grafana_secret],
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    max_containers=5,
    scaledown_window=10,
    timeout=120,
)
class FastCpuEngine:
    """optimized cpu inference with parallel execution.
    Grafana Cloud OTEL instrumentation for production observability."""
    
    @modal.enter()
    def setup(self):
        """initialize models and clients."""
        import httpx
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        import libsql_experimental as libsql
        import boto3
        from botocore.config import Config
        import zstandard
        
        # Initialize OpenTelemetry for Grafana Cloud
        self.tracer = None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            
            resource = Resource.create({"service.name": "nvyra-x-free"})
            provider = TracerProvider(resource=resource)
            
            # Grafana Cloud OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=f"{OTEL_ENDPOINT}/v1/traces",
                headers={"Authorization": "Basic MTQ4Mzc0NDpnbGNfZXlKdklqb2lNVFl6TURnNE1TSXNJbTRpT2lKdWRubHlZUzE0SWl3aWF5STZJbWd6WVZNNFJ6SjJRMWxST0dFd05qYzFRamd3VTBONFV5SXNJbTBpT25zaWNpSTZJbkJ5YjJRdFpYVXRibTl5ZEdndE1DSjlmUT09"},
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer("nvyra-x-free")
            print("✅ Grafana Cloud OTEL initialized")
        except Exception as e:
            print(f"⚠️ OTEL init failed (non-critical): {e}")
        
        print(f"initializing fast cpu engine... (OTEL: {'ENABLED' if self.tracer else 'DISABLED'})")
        
        # find gguf models
        self.qwen3_path = self._find_gguf("/models", "qwen3")
        self.disinfo_path = self._find_gguf("/models", "fake-news")
        self.reasoning_path = self._find_gguf("/models", "reasoning")
        
        # http client for search
        self.http = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
        self.robots_cache = RobotsCache()
        
        # embedding model
        try:
            self.embedder = SentenceTransformer(bge_model)
        except Exception:
            self.embedder = None
        
        # storage clients for cache lookup
        self.qc = QdrantClient(url=qdrant_url)
        self.db = libsql.connect(database=turso_url, auth_token=turso_token)
        self.cctx = zstandard.ZstdCompressor(level=3)
        self.dctx = zstandard.ZstdDecompressor()
        self.s3 = boto3.client(
            's3',
            endpoint_url=b2_endpoint,
            aws_access_key_id=b2_access_key,
            aws_secret_access_key=b2_secret_key,
            config=Config(max_pool_connections=20),
        )
        
        # rate limit tracking
        self.engine_usage = {e: deque(maxlen=100) for e in search_rate_limits}
        
        print(f"cpu engine ready - qwen3: {self.qwen3_path}, disinfo: {self.disinfo_path}")
    
    def _find_gguf(self, base: str, pattern: str) -> str:
        """find gguf file matching pattern."""
        p = Path(base)
        for f in p.rglob("*.gguf"):
            if pattern.lower() in f.name.lower():
                return str(f)
        return ""
    
    def _run_llama(self, model_path: str, prompt: str, max_tokens: int = 512) -> str:
        """run inference with llama.cpp."""
        if not model_path:
            return '{"error": "model not available"}'
        
        try:
            cmd = [
                "llama-cli",
                "-m", model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", "0.3",
                "--top-p", "0.9",
                "-ngl", "0",
                "--no-display-prompt",
                "-c", "4096",
                "-t", "8",
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception as e:
            return f'{{"error": "{str(e)[:100]}"}}'
    
    async def _search_cache(self, claim: str) -> Optional[Dict[str, Any]]:
        """search qdrant cache for similar claims."""
        try:
            if not self.embedder:
                return None
            
            # compute embedding
            embedding = self.embedder.encode(claim).tolist()
            
            # search qdrant
            results = self.qc.search(
                collection_name=qdrant_collection,
                query_vector=("dense", embedding),
                limit=3,
                score_threshold=0.85,
            )
            
            if not results:
                return None
            
            best = results[0]
            claim_id = best.payload.get("claim_id")
            
            # fetch from turso
            row = self.db.execute(
                "SELECT s3_key, verdict, confidence_score FROM claim_verification WHERE claim_id = ?",
                (claim_id,)
            ).fetchone()
            
            if not row:
                return None
            
            s3_key, verdict, confidence = row
            
            # fetch from backblaze if available
            content = {}
            if s3_key:
                try:
                    obj = self.s3.get_object(Bucket=b2_bucket, Key=s3_key)
                    content = json.loads(self.dctx.decompress(obj['Body'].read()))
                except Exception:
                    pass
            
            return {
                "cache_hit": True,
                "verdict": verdict,
                "confidence": confidence,
                "content": content,
                "score": best.score,
            }
        except Exception as e:
            print(f"cache search error: {e}")
            return None
    
    async def _check_robots(self, url: str) -> bool:
        """check robots.txt compliance."""
        try:
            parsed = urlparse(url)
            domain = f"{parsed.scheme}://{parsed.netloc}"
            
            parser = self.robots_cache.get(domain)
            if parser is None:
                parser = RobotFileParser()
                try:
                    resp = await self.http.get(f"{domain}/robots.txt", timeout=3.0)
                    if resp.status_code == 200:
                        parser.parse(resp.text.splitlines())
                    else:
                        parser.allow_all = True
                except Exception:
                    parser.allow_all = True
                self.robots_cache.set(domain, parser)
            
            return parser.can_fetch("NvyraBot/1.0", url)
        except Exception:
            return True
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """search duckduckgo."""
        results = []
        try:
            resp = await self.http.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "kl": "us-en"},
                headers={"User-Agent": "Mozilla/5.0 (compatible; NvyraBot/1.0)"},
            )
            
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "lxml")
                
                for r in soup.select(".result")[:5]:
                    title = r.select_one(".result__title a")
                    snippet = r.select_one(".result__snippet")
                    if title and snippet:
                        url = title.get("href", "")
                        if "/l/?uddg=" in url:
                            url = url.split("uddg=")[-1].split("&")[0]
                        results.append({
                            "url": html.unescape(url),
                            "title": title.get_text(strip=True),
                            "snippet": snippet.get_text(strip=True),
                        })
        except Exception:
            pass
        return results
    
    async def _search_bing(self, query: str) -> List[Dict[str, str]]:
        """search bing."""
        results = []
        try:
            resp = await self.http.get(
                "https://www.bing.com/search",
                params={"q": query, "count": "5"},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "lxml")
                
                for li in soup.select("li.b_algo")[:5]:
                    title = li.select_one("h2 a")
                    snippet = li.select_one(".b_caption p")
                    if title:
                        results.append({
                            "url": title.get("href", ""),
                            "title": title.get_text(strip=True),
                            "snippet": snippet.get_text(strip=True) if snippet else "",
                        })
        except Exception:
            pass
        return results
    
    async def _scrape_content(self, url: str) -> Optional[str]:
        """scrape and extract content."""
        if not await self._check_robots(url):
            return None
        
        try:
            resp = await self.http.get(url, timeout=8.0)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                return None
            
            import trafilatura
            extracted = trafilatura.extract(resp.text, include_comments=False, favor_precision=True)
            return extracted[:5000] if extracted else None
        except Exception:
            return None
    
    async def _multi_search(self, query: str) -> Tuple[str, List[str]]:
        """search multiple engines and scrape content."""
        all_results = []
        sources = []
        
        # search engines in parallel
        tasks = [
            self._search_duckduckgo(query),
            self._search_bing(query),
        ]
        
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result_list in results_lists:
            if isinstance(result_list, list):
                all_results.extend(result_list)
        
        # deduplicate
        seen = set()
        unique = []
        for r in all_results:
            url_hash = hashlib.md5(r["url"].encode()).hexdigest()
            if url_hash not in seen:
                seen.add(url_hash)
                unique.append(r)
        
        # scrape top results
        evidence = ""
        scrape_tasks = [self._scrape_content(r["url"]) for r in unique[:3]]
        scraped = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        
        for i, content in enumerate(scraped):
            if isinstance(content, str) and content:
                evidence += f"\n\nsource: {unique[i]['title']}\nurl: {unique[i]['url']}\n{content[:2000]}"
                sources.append(unique[i]['url'])
        
        return evidence, sources
    
    def _run_claim_extraction(self, text: str) -> Dict[str, Any]:
        """extract claims and decide search strategy with qwen3."""
        prompt = f"""<|im_start|>system
You are nvyra-x claim analyzer. Analyze the input and output JSON:
{{"needs_search": true/false, "search_query": "...", "claim_type": "factual/opinion/greeting", "is_simple": true/false}}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
        
        raw = self._run_llama(self.qwen3_path, prompt, 128)
        
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        
        return {"needs_search": True, "search_query": text[:100], "claim_type": "factual", "is_simple": False}
    
    def _run_disinfo_check(self, text: str, context: str = "") -> Dict[str, Any]:
        """run disinformation detection."""
        prompt = f"""Analyze for disinformation patterns. Output JSON with disinfo_score (0-1).
Text: {text}
Context: {context[:1000] if context else 'none'}

JSON:"""
        
        raw = self._run_llama(self.disinfo_path or self.qwen3_path, prompt, 256)
        
        try:
            match = re.search(r'(?:score|disinfo)[:\s]*([0-9.]+)', raw.lower())
            score = float(match.group(1)) if match else 0.5
            return {"disinfo_score": min(max(score, 0.0), 1.0), "analysis": raw[:200]}
        except Exception:
            return {"disinfo_score": 0.5, "analysis": ""}
    
    def _run_fact_check(self, claim: str, evidence: str) -> Dict[str, Any]:
        """fact check with qwen3."""
        prompt = f"""<|im_start|>system
You are an expert fact-checker. Output JSON: {{"verdict": "true/false/partially_true/unverifiable/misleading", "confidence": 0.0-1.0, "reasoning": "..."}}<|im_end|>
<|im_start|>user
Claim: {claim}
Evidence: {evidence[:4000]}<|im_end|>
<|im_start|>assistant
"""
        
        raw = self._run_llama(self.qwen3_path, prompt, 512)
        
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if "verdict" in result:
                    return result
        except Exception:
            pass
        
        return {"verdict": "unverifiable", "confidence": 0.5, "reasoning": raw[:300]}
    
    def _run_reasoning(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """combine outputs with reasoning model."""
        fc = inputs.get("factcheck", {})
        dis = inputs.get("disinfo", {})
        
        # if reasoning model available
        if self.reasoning_path:
            prompt = f"""Combine these analyses:
Fact-check: {json.dumps(fc)}
Disinfo: {json.dumps(dis)}
Output: {{"verdict": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""
            
            raw = self._run_llama(self.reasoning_path, prompt, 256)
            try:
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except Exception:
                pass
        
        # fallback combination
        fc_conf = fc.get("confidence", 0.5)
        fc_verdict = fc.get("verdict", "unverifiable")
        disinfo_score = dis.get("disinfo_score", 0.5)
        
        verdict_map = {"true": 0.0, "false": 1.0, "partially_true": 0.4, "misleading": 0.7, "unverifiable": 0.5}
        combined = (verdict_map.get(fc_verdict, 0.5) * 0.5) + (disinfo_score * 0.5)
        
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
            "reasoning": f"factcheck: {fc_verdict}, disinfo: {disinfo_score:.2f}",
        }
    
    @modal.method()
    async def verify(self, request: VerificationRequest) -> VerificationResult:
        """main verification pipeline - linear execution as per pipeline.md."""
        start = time.perf_counter()
        req_id = request.request_id or uuid.uuid4().hex[:16]
        
        # step 1: claim extraction and strategy
        plan = await asyncio.to_thread(self._run_claim_extraction, request.claim)
        
        # handle simple queries
        if plan.get("is_simple") or plan.get("claim_type") == "greeting":
            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                request_id=req_id,
                claim=request.claim,
                verdict=Verdict.TRUE,
                confidence_score=1.0,
                falsity_score=0.0,
                reasoning="hello! i'm nvyra-x, your fact-checking assistant.",
                latency_ms=latency,
            )
        
        # step 2: check cache first
        cache_result = await self._search_cache(request.claim)
        if cache_result and cache_result.get("cache_hit"):
            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                request_id=req_id,
                claim=request.claim,
                verdict=Verdict(cache_result.get("verdict", "unverifiable")),
                confidence_score=cache_result.get("confidence", 0.8),
                falsity_score=1 - cache_result.get("confidence", 0.8),
                reasoning="retrieved from cache",
                latency_ms=latency,
                cache_hit=True,
            )
        
        # step 3: parallel - search and disinfo
        evidence = request.context or ""
        sources = []
        
        if plan.get("needs_search", True) and not evidence:
            query = plan.get("search_query", request.claim[:100])
            
            # run search and disinfo in parallel
            search_task = asyncio.create_task(self._multi_search(query))
            disinfo_task = asyncio.to_thread(self._run_disinfo_check, request.claim, "")
            
            (new_evidence, sources), disinfo_result = await asyncio.gather(search_task, disinfo_task)
            evidence = new_evidence
        else:
            disinfo_result = await asyncio.to_thread(self._run_disinfo_check, request.claim, evidence)
        
        # step 4: fact check with evidence
        factcheck_result = await asyncio.to_thread(self._run_fact_check, request.claim, evidence)
        
        # step 5: combine with reasoning
        combined = await asyncio.to_thread(self._run_reasoning, {
            "factcheck": factcheck_result,
            "disinfo": disinfo_result,
        })
        
        latency = (time.perf_counter() - start) * 1000
        
        return VerificationResult(
            request_id=req_id,
            claim=request.claim,
            verdict=Verdict(combined.get("verdict", "unverifiable")),
            confidence_score=combined.get("confidence", 0.5),
            falsity_score=1 - combined.get("confidence", 0.5),
            reasoning=combined.get("reasoning", ""),
            sources_used=sources[:5],
            latency_ms=latency,
        )


@app.function(image=modal.Image.debian_slim().pip_install("pydantic", "fastapi"))
@modal.fastapi_endpoint(method="POST")
async def verify_claim(request: VerificationRequest) -> VerificationResult:
    """public api endpoint."""
    engine = FastCpuEngine()
    return await engine.verify.remote.aio(request)


@app.local_entrypoint()
def main():
    """test free tier pipeline."""
    print("testing nvyra-x free pipeline...")
    
    tests = [
        "hello, who are you?",
        "the earth is flat.",
        "vaccines are safe and effective.",
    ]
    
    engine = FastCpuEngine()
    for text in tests:
        print(f"\ninput: {text}")
        result = engine.verify.remote(VerificationRequest(claim=text))
        print(f"  verdict: {result.verdict.value}")
        print(f"  confidence: {result.confidence_score:.2%}")
        print(f"  cache_hit: {result.cache_hit}")
        print(f"  latency: {result.latency_ms:.0f}ms")
