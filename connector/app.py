import json
import os
import pathlib
import shlex
import subprocess
import uuid

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()
API_KEY = os.getenv("X_API_KEY", "dev-key")

class RunRequest(BaseModel):
    audio_url: str | None = None
    params: dict = {}

def run_pipeline(audio_url: str | None, params: dict, outdir: str):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    # TODO: swap in your real CLI; this is a stub
    cmd = f'python transcribe.py --audio "{audio_url or ""}" --outdir "{outdir}"'
    if params:
        cmd += " --params " + shlex.quote(json.dumps(params))
    subprocess.run(cmd, shell=True, check=False)

def ensure_auth(x_api_key: str | None):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/run")
def run_job(req: RunRequest, bg: BackgroundTasks, x_api_key: str | None = Header(None, alias="x-api-key")):
    ensure_auth(x_api_key)
    job_id = str(uuid.uuid4())
    outdir = str(pathlib.Path("runs") / job_id)
    bg.add_task(run_pipeline, req.audio_url, req.params, outdir)
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def status(job_id: str, x_api_key: str | None = Header(None, alias="x-api-key")):
    ensure_auth(x_api_key)
    outdir = pathlib.Path("runs") / job_id
    done = (outdir / "summary.html").exists()
    return {"job_id": job_id, "status": "done" if done else "running"}

@app.get("/results/{job_id}")
def results(job_id: str, x_api_key: str | None = Header(None, alias="x-api-key")):
    ensure_auth(x_api_key)
    base = f"/static/{job_id}"
    return {"job_id": job_id, "files": {
        "csv": f"{base}/diarized_transcript_with_emotion.csv",
        "summary_html": f"{base}/summary.html",
        "speakers": f"{base}/speakers_summary.csv"
    }}
