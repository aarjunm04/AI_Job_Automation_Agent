from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Auto Apply Service")

class ApplyRequest(BaseModel):
    job_url: str
    resume_filename: str

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "auto_apply"}

@app.post("/apply")
def apply_to_job(request: ApplyRequest):
    # Full logic is in auto_apply/modules, this is a stub HTTP entry point
    return {"status": "queued"}
