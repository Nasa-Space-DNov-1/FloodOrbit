from fastapi import FastAPI, HTTPException
from app.schemas import PipelineRunRequest, PipelineResponse
from app.pipeline import execute_16_stage_pipeline

app = FastAPI(
    title="FloodOrbit: NISAR Flash-Flood Detection Backend",
    version="1.0.0",
    description="Production-grade 16-stage backend for processing NASA-ISRO NISAR L2 GCOV products."
)

@app.post("/api/v1/run-pipeline", response_model=PipelineResponse)
def run_flash_flood_pipeline(payload: PipelineRunRequest):
    try:
        result = execute_16_stage_pipeline(payload)
        return PipelineResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))