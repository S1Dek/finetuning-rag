#!/bin/bash
source ai-env/bin/activate
cd rag
uvicorn api:app --host 0.0.0.0 --port 8000