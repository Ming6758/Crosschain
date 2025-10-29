# main.py
from fastapi import FastAPI, Request
from orchestrator.langgraph_runner import create_graph
from typing import Dict, Any
import uvicorn
import logging
from datetime import datetime

app = FastAPI()
graph = create_graph()                     # compiled once at start-up

# Structured logging (JSON lines for prod)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

@app.post("/process")
async def process_alert(request: Request) -> Dict[str, Any]:
    """
    Real-time entry point – receives a raw alert, runs the LangGraph pipeline,
    and returns a full incident report.
    """
    try:
        body: Dict[str, Any] = await request.json()
        alert_type = body.get("alertType", "Unknown")
        logger.info("Received alert", extra={"alert_type": alert_type, "payload": body})

        # guarantee a timestamp
        if "timestamp" not in body:
            body["timestamp"] = int(datetime.now().timestamp())

        start = datetime.now()
        output = graph.invoke(body)                     # LangGraph is sync – fine for <100 ms
        elapsed = (datetime.now() - start).total_seconds()

        # enrich with processing metadata
        output["processing_metadata"] = {
            "time_seconds": round(elapsed, 3),
            "agents_executed": [k for k in output if not k.startswith("_")],
            "success": True,
        }

        logger.info("Alert processed successfully", extra={"elapsed": elapsed})
        return output

    except Exception as exc:
        logger.error("Alert processing failed", exc_info=True)
        return {
            "error": str(exc),
            "processing_metadata": {"success": False, "error_details": str(exc)},
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)   # 4 workers for real-time load
