# main.py
from fastapi import FastAPI, Request
from orchestrator.langgraph_runner import create_graph
from typing import Dict, Any
import uvicorn
import logging
from datetime import datetime

app = FastAPI()
graph = create_graph()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/process")
async def process_alert(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
        logger.info(f"Processing alert: {body.get('alertType', 'Unknown')}")
        
        # Add timestamp if not present
        if "timestamp" not in body:
            body["timestamp"] = int(datetime.now().timestamp())
        
        # Execute the graph (LangGraph handles sync/async internally)
        start_time = datetime.now()
        output = graph.invoke(body)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add processing metadata
        output["processing_metadata"] = {
            "time_seconds": processing_time,
            "agents_executed": list(output.keys()),
            "success": True
        }
        
        return output
    
    except Exception as e:
        logger.error(f"Error processing alert: {str(e)}")
        return {
            "error": str(e),
            "processing_metadata": {
                "success": False,
                "error_details": str(e)
            }
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
