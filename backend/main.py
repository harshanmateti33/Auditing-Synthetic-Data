# backend/main.py
import io
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

import metrics
from nemo_orchestrator import run_nemo_orchestration

# Graceful GPU import fallback
try:
    import cudf
    GPU_ACCELERATED = True
except ImportError:
    import pandas as cudf
    GPU_ACCELERATED = False

app = FastAPI(title="TrustForge NVIDIA Stack Python Gateway", version="1.0.0")

# Enable CORS for Next.js API integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    csvText: str
    datasetName: str

class RepairRequest(BaseModel):
    csvText: str
    actions: List[str]
    metrics: Dict[str, Any]

@app.post("/analyze")
async def analyze_endpoint(payload: AnalyzeRequest):
    try:
        # Load CSV using cuDF / Pandas on GPU or CPU
        df = cudf.read_csv(io.StringIO(payload.csvText))
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")
            
        result = run_nemo_orchestration(df, payload.datasetName)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/repair")
async def repair_endpoint(payload: RepairRequest):
    try:
        # Load CSV using cuDF / Pandas on GPU or CPU
        df = cudf.read_csv(io.StringIO(payload.csvText))
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
            
        repaired_df, repair_log = metrics.repair_dataset(df, payload.actions, payload.metrics)
        
        # Convert repaired DataFrame back to CSV string
        pdf = repaired_df.to_pandas() if hasattr(repaired_df, 'to_pandas') else repaired_df
        csv_buffer = io.StringIO()
        pdf.to_csv(csv_buffer, index=False)
        repaired_csv_text = csv_buffer.getvalue()
        
        # Evaluate trust scores on the newly repaired dataset
        result = run_nemo_orchestration(repaired_df, "repaired.csv")
        result["repaired_csv"] = repaired_csv_text
        result["repair_log"] = repair_log
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "TrustForge Gateway",
        "nvidia_gpu_acceleration": GPU_ACCELERATED
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
