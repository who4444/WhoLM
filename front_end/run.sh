#!/bin/bash
# Run WhoLM Frontend

echo "Starting WhoLM Frontend..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
pip install -r requirements.txt

# Run Streamlit app
echo "Launching Streamlit app..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0