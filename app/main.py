# -*- coding: utf-8 -*-
"""
AI Deployment Architecture Comparator — v3.1
- FastAPI backend for quick comparison and load benchmarking
- Works with a Triton (Proposed) endpoint and a Baseline FastAPI/Torch endpoint
- Health checks and CSV/JSON-friendly outputs
"""

import asyncio, json, os, time, base64, io, mimetypes
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np, uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Auto-install minimal deps on first run
def _ensure(pkgs: List[str]):
    import importlib, subprocess, sys
    for p in pkgs:
        try:
            importlib.import_module(p)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--root-user-action=ignore", p])

_ensure(["httpx", "python-multipart", "tritonclient[http]", "pillow"])

import httpx
import tritonclient.http as triton_http
from PIL import Image


app = FastAPI(title="AI Deployment Architecture Comparator", version="3.1")

# CORS: keep permissive for demos; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"])
)

# ----------- Defaults (env-overridable) -----------
DEFAULT_TRITON_URL = os.environ.get("TRITON_URL", "192.168.130.206:8000")
DEFAULT_MODEL_NAME = os.environ.get("TRITON_MODEL", "gender_ensemble")
DEFAULT_TRITON_INPUT_NAME = os.environ.get("TRITON_INPUT_NAME", "RAW_IMAGE")
DEFAULT_TRITON_OUTPUTS = json.loads(os.environ.get("TRITON_OUTPUTS", '["MAN_PROB","WOMAN_PROB","LABEL"]'))
DEFAULT_MAX_BATCH = int(os.environ.get("TRITON_MAX_BATCH", "32"))
DEFAULT_BASE_API = os.environ.get("BASE_API_URL", "http://192.168.130.65:5050/upload/predict_gender")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "60"))

# Optional diagram paths
BASELINE_DIAGRAM_PATH = os.environ.get(
    "BASELINE_DIAGRAM_PATH",
    "/home/roknabadi/projects/SE_FOR_AI/compare_ui/base_container_c4_diagram.png",
)
PROPOSED_DIAGRAM_PATH = os.environ.get(
    "PROPOSED_DIAGRAM_PATH",
    "/home/roknabadi/projects/SE_FOR_AI/compare_ui/my_container_c4_arch.png",
)


@dataclass
class Scenario:
    """Load test scenario descriptor."""
    name: str
    batch_size: int
    concurrency: int
    total_images: int


# ---------- helpers ----------
def _now() -> float: return time.perf_counter()

def _percentiles(v: List[float], ps=(50, 90, 95, 99)) -> Dict[str, float]:
    if not v: return {f"p{p}": 0.0 for p in ps}
    arr = np.array(v); return {f"p{p}": float(np.percentile(arr, p)) for p in ps}

def _summary(v: List[float]) -> Dict[str, float]:
    if not v: return dict(count=0, avg=0.0, min=0.0, max=0.0, **_percentiles([]))
    return dict(count=len(v), avg=float(mean(v)), min=float(min(v)), max=float(max(v)), **_percentiles(v))

def _bucket_by_second(timestamps: List[float]) -> Dict[int, int]:
    b=defaultdict(int)
    for ts in timestamps: b[int(ts)] += 1
    return dict(sorted(b.items()))

def _dominant(man: float, woman: float) -> str: return "Man" if man>=woman else "Woman"

def _pad_batch(raw: List[bytes]) -> np.ndarray:
    """Pad variable-length image bytes to a 2D UINT8 batch tensor (B, max_len)."""
    arrs=[np.frombuffer(b, dtype=np.uint8) for b in raw]
    m=max(a.shape[0] for a in arrs)
    out=np.zeros((len(arrs), m), dtype=np.uint8)
    for i,a in enumerate(arrs): out[i,:a.shape[0]] = a
    return out

def _ensure_http(u: str) -> str:
    return u if u.startswith("http://") or u.startswith("https://") else f"http://{u}"

def _thumb_data_url(b: bytes, max_side=256) -> str:
    """Generate a small base64 thumbnail for UI previews (safe fallback for non-images)."""
    try:
        im = Image.open(io.BytesIO(b)).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=80)
        enc = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{enc}"
    except Exception:
        enc = base64.b64encode(b[:100_000]).decode("ascii")
        return f"data:application/octet-stream;base64,{enc}"


# ---------- inference ----------
def triton_infer_bytes_batch(
    triton_url: str,
    model_name: str,
    input_name: str,
    output_names: List[str],
    raw_images: List[bytes],
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Perform a single Triton inference call with a padded UINT8 batch.
    Return:
      - predictions list (per image)
      - total batch latency in seconds
    """
    if not raw_images: return [], 0.0

    client = triton_http.InferenceServerClient(url=triton_url, ssl=False, network_timeout=timeout_s)
    batch = _pad_batch(raw_images)

    inp = triton_http.InferInput(name=input_name, shape=batch.shape, datatype="UINT8")
    inp.set_data_from_numpy(batch)

    req_outs = [triton_http.InferRequestedOutput(n) for n in output_names]

    t0=_now(); resp=client.infer(model_name=model_name, inputs=[inp], outputs=req_outs); t1=_now()

    man = resp.as_numpy(output_names[0])
    woman = resp.as_numpy(output_names[1])
    labels = resp.as_numpy(output_names[2]) if len(output_names)>2 else None

    if man is None or woman is None:
        got = [o["name"] for o in resp.get_response().get("outputs", [])]
        raise RuntimeError(f"Triton missing required outputs {output_names[:2]} (got: {got})")

    preds=[]
    for i in range(len(raw_images)):
        mp=float(man[i][0]); wp=float(woman[i][0])
        lab = labels[i][0].decode("utf-8") if (labels is not None and labels[i][0] is not None) else _dominant(mp,wp)
        preds.append(dict(MAN_PROB=mp, WOMAN_PROB=wp, LABEL=lab, dominant=_dominant(mp,wp)))
    return preds, (t1-t0)


# ------------------ Pages ------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    tpl = jinja_env.get_template("index.html")
    return HTMLResponse(tpl.render(
        active_tab="compare",
        default_triton_url=DEFAULT_TRITON_URL,
        default_model_name=DEFAULT_MODEL_NAME,
        default_input_name=DEFAULT_TRITON_INPUT_NAME,
        default_outputs=",".join(DEFAULT_TRITON_OUTPUTS),
        default_max_batch=DEFAULT_MAX_BATCH,
        default_base_api=DEFAULT_BASE_API,
        request_timeout=REQUEST_TIMEOUT,
    ))

@app.get("/design", response_class=HTMLResponse)
async def design():
    tpl = jinja_env.get_template("design.html")
    baseline_exists = os.path.exists(BASELINE_DIAGRAM_PATH)
    proposed_exists = os.path.exists(PROPOSED_DIAGRAM_PATH)
    return HTMLResponse(tpl.render(
        active_tab="design",
        baseline_exists=baseline_exists,
        proposed_exists=proposed_exists,
        baseline_path=BASELINE_DIAGRAM_PATH,
        proposed_path=PROPOSED_DIAGRAM_PATH,
    ))


# ------------------ Diagram files ------------------
def _serve_diagram_file(path: str) -> FileResponse:
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Diagram not found on server: {path}")
    media_type = mimetypes.guess_type(path)[0] or "image/png"
    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))

@app.get("/diagram/baseline")
def diagram_baseline(): return _serve_diagram_file(BASELINE_DIAGRAM_PATH)

@app.get("/diagram/proposed")
def diagram_proposed(): return _serve_diagram_file(PROPOSED_DIAGRAM_PATH)


# ------------------ Health checks ------------------
@app.get("/api/health/triton")
async def health_triton(url: str, timeout_s: float = REQUEST_TIMEOUT):
    try:
        client = triton_http.InferenceServerClient(url=url, ssl=False, network_timeout=timeout_s)
        return {"ok": True, "ready": bool(client.is_server_ready()), "live": bool(client.is_server_live())}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)

@app.get("/api/health/baseline")
async def health_baseline(url: str, timeout_s: float = REQUEST_TIMEOUT):
    url = _ensure_http(url)
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.head(url)
        return {"ok": (200 <= r.status_code < 500), "status_code": r.status_code}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)


# ------------------ API: predict & benchmark ------------------
@app.post("/api/predict")
async def api_predict(
    files: List[UploadFile] = File(...),
    triton_url: str = Form(DEFAULT_TRITON_URL),
    model_name: str = Form(DEFAULT_MODEL_NAME),
    triton_input_name: str = Form(DEFAULT_TRITON_INPUT_NAME),
    triton_outputs_csv: str = Form(",".join(DEFAULT_TRITON_OUTPUTS)),
    base_api_url: str = Form(DEFAULT_BASE_API),
    max_batch: int = Form(DEFAULT_MAX_BATCH),
    timeout_s: float = Form(REQUEST_TIMEOUT),
):
    """Quick comparison: Triton batched vs. baseline per-image (no charts in UI)."""
    outs = [s.strip() for s in triton_outputs_csv.split(",") if s.strip()]
    if len(outs) < 2:
        return JSONResponse({"status":"error","error":"Need outputs: MAN_PROB,WOMAN_PROB,(LABEL optional)"}, status_code=400)

    raw = [await f.read() for f in files]
    if not raw: raise HTTPException(400, "No images uploaded.")
    thumbs = [ _thumb_data_url(b) for b in raw[:10] ]

    # Proposed (Triton)
    tri_preds, tri_lat_ms, tri_img_ts, tri_req_ts = [], [], [], []
    tri_errors, tri_errs, tri_batch_sizes = 0, [], []
    for i in range(0, len(raw), max_batch):
        batch = raw[i:i+max_batch]
        try:
            preds, lat_s = await asyncio.to_thread(
                triton_infer_bytes_batch, triton_url, model_name, triton_input_name, outs, batch, timeout_s
            )
            t_end = _now()
            tri_preds.extend(preds)
            tri_lat_ms.append(lat_s*1000.0)
            tri_req_ts.append(t_end)
            tri_img_ts.extend([t_end]*len(batch))
            tri_batch_sizes.append(len(batch))
        except Exception as e:
            tri_errors += len(batch); tri_errs.append(str(e))

    # Baseline (per-image)
    base_api_url = _ensure_http(base_api_url)
    base_preds, base_lat_ms, base_img_ts = [], [], []
    base_errors, base_errs = 0, []
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        sem = asyncio.Semaphore(64)
        async def _one(img: bytes):
            nonlocal base_errors
            t0=_now()
            try:
                async with sem:
                    r = await client.post(base_api_url, files={"file":("image.jpg", img, "image/jpeg")})
                r.raise_for_status()
                j=r.json(); res=j["result"][0]
                man=float(res["gender"]["Man"]); woman=float(res["gender"]["Woman"])
                base_preds.append(dict(MAN_PROB=man, WOMAN_PROB=woman, LABEL=res.get("dominant_gender"), dominant=_dominant(man,woman)))
                t1=_now(); base_lat_ms.append((t1-t0)*1000.0); base_img_ts.append(t1)
            except Exception as e:
                base_errors+=1; base_errs.append(str(e))
        await asyncio.gather(*[_one(b) for b in raw], return_exceptions=True)

    # Basic summaries (still computed, even if UI doesn't chart them)
    def _rate(count:int, ts:List[float]) -> float:
        if not ts or count==0: return 0.0
        dur = max(ts) - min(ts)
        return count / max(dur, 1e-6)
    tri_ips = _rate(len(tri_img_ts), tri_img_ts)
    tri_qps = _rate(len(tri_req_ts), tri_req_ts)
    base_ips = _rate(len(base_img_ts), base_img_ts)
    base_qps = base_ips

    tri_per_image_lat = [(l/bs) for l,bs in zip(tri_lat_ms, tri_batch_sizes)] if tri_batch_sizes else []
    tri_per_image_summary = _summary(tri_per_image_lat)

    n=len(raw)
    def pad(L, fill=None): return (L+[fill]*max(0,n-len(L)))[:n]
    tri_preds = pad(tri_preds,{}); base_preds = pad(base_preds,{})

    return JSONResponse(dict(
        status="ok",
        counts=dict(images=n, triton_errors=tri_errors, base_errors=base_errors),
        errors_detail=dict(triton=tri_errs[:5], base=base_errs[:5]),
        predictions=[dict(index=i, img=(thumbs[i] if i < len(thumbs) else None), triton=tri_preds[i], base=base_preds[i]) for i in range(n)],
        metrics=dict(  # kept for CSV/JSON export; UI won't plot them
            triton=dict(batch_latency_ms=_summary(tri_lat_ms), approx_per_image_latency_ms=tri_per_image_summary, ips=tri_ips, qps=tri_qps),
            base=dict(per_request_latency_ms=_summary(base_lat_ms), ips=base_ips, qps=base_qps),
        ),
        speedups=dict(
            ips = (tri_ips / base_ips) if base_ips>0 else None,
            qps = (tri_qps / base_qps) if base_qps>0 else None,
            approx_p50_latency = ( (np.percentile(base_lat_ms,50)) / max(tri_per_image_summary.get("p50",1e-6), 1e-6) ) if base_lat_ms else None
        )
    ))


@app.post("/api/benchmark")
async def api_benchmark(files: List[UploadFile] = File(...), config_json: str = Form(...)):
    """Load tests with multiple scenarios (charts shown in UI)."""
    cfg=json.loads(config_json)
    triton_url=cfg.get("triton_url",DEFAULT_TRITON_URL)
    model_name=cfg.get("model_name",DEFAULT_MODEL_NAME)
    triton_input_name=cfg.get("triton_input_name",DEFAULT_TRITON_INPUT_NAME)
    triton_outputs=cfg.get("triton_outputs",DEFAULT_TRITON_OUTPUTS)
    base_api_url=_ensure_http(cfg.get("base_api_url",DEFAULT_BASE_API))
    max_batch=int(cfg.get("max_batch",DEFAULT_MAX_BATCH))
    timeout_s=float(cfg.get("timeout_s",REQUEST_TIMEOUT))
    scenarios=[Scenario(**s) for s in cfg["scenarios"]]
    raw=[await f.read() for f in files]
    if not raw: raise HTTPException(400,"No images uploaded.")

    def stream(total:int):
        if total<=len(raw): return raw[:total]
        reps=(total+len(raw)-1)//len(raw); return (raw*reps)[:total]

    async def run(sc:Scenario) -> Dict[str,Any]:
        tri_lat, tri_img_ts, tri_req_ts, tri_err, tri_batch_sizes = [], [], [], 0, []
        base_lat, base_img_ts, base_err = [], [], 0
        imgs = stream(sc.total_images)

        async def tri_worker(batches: List[List[bytes]]):
            nonlocal tri_err
            for b in batches:
                try:
                    _, lat_s = await asyncio.to_thread(
                        triton_infer_bytes_batch, triton_url, model_name, triton_input_name, triton_outputs, b[:max_batch], timeout_s
                    )
                    t1=_now()
                    tri_lat.append(lat_s*1000.0); tri_req_ts.append(t1)
                    tri_img_ts.extend([t1]*len(b)); tri_batch_sizes.append(len(b))
                except Exception: tri_err += len(b)

        async def base_worker(images: List[bytes]):
            nonlocal base_err
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                sem=asyncio.Semaphore(sc.batch_size)
                async def one(img):
                    nonlocal base_err
                    t0=_now()
                    try:
                        async with sem:
                            r=await client.post(base_api_url, files={"file":("image.jpg",img,"image/jpeg")})
                        r.raise_for_status(); _=r.json()
                        t1=_now(); base_lat.append((t1-t0)*1000.0); base_img_ts.append(t1)
                    except Exception: base_err += 1
                await asyncio.gather(*[one(b) for b in images])

        batches=[imgs[i:i+sc.batch_size] for i in range(0,len(imgs), sc.batch_size)]
        tri_batches=[batches[i::sc.concurrency] for i in range(sc.concurrency)]
        base_images=[imgs[i::sc.concurrency] for i in range(sc.concurrency)]
        await asyncio.gather(*([tri_worker(tri_batches[i]) for i in range(sc.concurrency)] +
                               [base_worker(base_images[i]) for i in range(sc.concurrency)]))

        def _rate(count:int, ts:List[float]) -> float:
            if not ts or count==0: return 0.0
            dur=max(ts)-min(ts); return count/max(dur,1e-6)

        tri_ips = _rate(len(tri_img_ts), tri_img_ts)
        tri_qps = _rate(len(tri_req_ts), tri_req_ts)
        base_ips = _rate(len(base_img_ts), base_img_ts)
        base_qps = base_ips

        tri_per_img_lat = [(l/bs) for l,bs in zip(tri_lat, tri_batch_sizes)] if tri_batch_sizes else []
        return dict(
            name=sc.name,
            config=dict(batch_size=sc.batch_size, concurrency=sc.concurrency, total_images=sc.total_images),
            triton=dict(
                latency_ms=_summary(tri_lat),
                approx_per_image_latency_ms=_summary(tri_per_img_lat),
                ips=tri_ips, qps=tri_qps, errors=tri_err,
                img_buckets=_bucket_by_second(tri_img_ts), req_buckets=_bucket_by_second(tri_req_ts)),
            base=dict(
                latency_ms=_summary(base_lat),
                ips=base_ips, qps=base_qps, errors=base_err,
                img_buckets=_bucket_by_second(base_img_ts), req_buckets=_bucket_by_second(base_img_ts)),
            speedups=dict(
                ips=(tri_ips/base_ips) if base_ips>0 else None,
                qps=(tri_qps/base_qps) if base_qps>0 else None,
                approx_p50_latency=( (np.percentile(base_lat,50)/max(np.percentile(tri_per_img_lat,50),1e-6)) if (base_lat and tri_per_img_lat) else None )
            )
        )

    out = dict(status="ok", scenarios=[])
    for sc in scenarios:
        out["scenarios"].append(await run(sc))
    return JSONResponse(out)


@app.get("/healthz")
def health(): return {"ok": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "5056")), reload=True)
