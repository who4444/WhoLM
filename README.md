# WhoLM - Multimodal Video & Document QA System

A sophisticated Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over videos and documents using multimodal embeddings, hybrid search, and generative AI.

## 🎯 Features

- **Video Question Answering**
  - YouTube URL support and local video uploads
  - AI-powered Q&A with timestamp citations
  - Frame extraction and visual understanding
  - Audio transcription and processing

- **Document Processing**
  - Support for PDF, DOCX, and TXT files
  - Intelligent document chunking and embedding
  - Context-aware retrieval

- **Advanced RAG Pipeline**
  - Hybrid retrieval combining BM25 (lexical) and dense embeddings
  - Reranking for improved relevance
  - Conversation memory for context-aware interactions
  - Multi-turn chat with session management

- **Multimodal Understanding**
  - Visual embeddings using CLIP (ViT-B-16-SigLIP)
  - Text embeddings with BGE models
  - Combined frame and text analysis

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **Vector DB**: Qdrant
- **LLM**: Google Generative AI (Gemini)
- **Processing**: OpenCV, Torch, Transformers
- **Storage**: Supabase

### Frontend
- **Framework**: Streamlit
- **HTTP Client**: Requests

### Key Libraries
- `yt-dlp` - YouTube video downloading
- `FlagEmbedding` - Text embeddings
- `open-clip-torch` - Image embeddings
- `stable-ts` - Audio transcription
- `PyMuPDF` / `python-docx` - Document parsing
- `rank-bm25` - BM25 ranking

## 📋 Prerequisites

- Python 3.8+
- Docker & Docker Compose (for containerized deployment)
- Qdrant instance running
- Google API key (for Gemini)
- Supabase account (for file storage)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd VideoQAAI
```

### 2. Backend Setup

```bash
cd back_end

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd front_end

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the `back_end` directory:

```env
# Qdrant Configuration
QDRANT_URL=http://localhost:6333

# Google Generative AI
GOOGLE_API_KEY=your_google_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_BUCKET_NAME=video-qa-bucket
```

### 5. Start Qdrant (if not running via Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -e QDRANT_API_KEY=your_api_key \
  qdrant/qdrant:latest
```

## ▶️ Running the Application

### Using Docker Compose (Recommended)

```bash
cd back_end
docker-compose up --build
```

### Manual Setup

**Terminal 1 - Backend Server:**
```bash
cd back_end
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend Application:**
```bash
cd front_end
source venv/bin/activate
streamlit run app.py
```

The frontend will be available at `http://localhost:8501` and the API at `http://localhost:8000`.

## 📚 Usage

### Video QA
1. Open the Streamlit app
2. Enter a YouTube URL or upload a local video file
3. Wait for processing (frame extraction, transcription, embedding)
4. Ask questions about the video content
5. Receive answers with timestamp citations

### Document QA
1. Upload PDF, DOCX, or TXT files
2. Files are processed and embedded
3. Query the documents using natural language
4. Get contextual answers based on document content

### Chat Interface
- Maintains conversation history within session
- Context-aware responses using previous interactions
- Clear session to start fresh conversation

## 📁 Project Structure

```
VideoQAAI/
├── back_end/
│   ├── api/                    # FastAPI routes and models
│   ├── config/                 # Configuration management
│   ├── database/               # Qdrant and Postgres connections
│   ├── ingestion/              # Video/document processing
│   │   ├── embeddings/         # CLIP and text embeddings
│   │   └── processing/         # Audio, frame, document processing
│   ├── services/               # Business logic
│   │   ├── chatbot/            # LLM and memory management
│   │   ├── rag/                # RAG pipeline
│   │   └── utils/              # Utilities
│   ├── main.py                 # Application entry point
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile              # Container configuration
├── front_end/
│   ├── app.py                  # Streamlit main interface
│   ├── video_player.py         # Video playback component
│   └── requirements.txt        # Frontend dependencies
├── qdrant_storage/             # Vector database data
└── README.md                   # This file
```

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `POST /upload/get-url` - Get presigned URL for file upload
- `POST /upload/youtube` - Process YouTube video
- `POST /upload/document` - Process document upload
- `POST /chat` - Send query and get response
- `GET /sessions/{session_id}` - Get session chat history

## ⚙️ Configuration

Key configuration options in `back_end/config/config.py`:

```python
BATCH_SIZE = 64              # Processing batch size
WORKER_NUM = 4              # Number of workers
CLIP_MODEL_NAME = 'ViT-B-16-SigLIP-512'  # Vision model
QDRANT_TEXT_EMBEDDING_DIM = 1024         # Text embedding dimension
QDRANT_FRAME_EMBEDDING_DIM = 768         # Frame embedding dimension
RAG_BM25_WEIGHT = 0.5       # BM25 retrieval weight
RAG_DENSE_WEIGHT = 0.5      # Dense retrieval weight
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🐛 Known Issues

See [bugs 1 create signed upload.txt](bugs%201%20create%20signed%20upload.txt) for known issues and planned fixes.

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the development team.

