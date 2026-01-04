#!/bin/bash
# Run WhoLM Backend API

echo "Starting WhoLM Backend API..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
pip install -r requirements.txt

# Run FastAPI server
echo "Launching FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload