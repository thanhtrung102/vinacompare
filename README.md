# VinaCompare: Vietnamese Technical Q&A RAG System

Comprehensive comparison of small Vietnamese language models in production RAG systems.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (recommended for LLM inference)

### 1. Setup Environment

```bash
# Navigate to project
cd D:/vinacompare

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Infrastructure

```bash
# Start all Docker services
docker-compose up -d

# Wait for services to be healthy (2-3 minutes)
docker-compose ps

# Check Milvus health
curl http://localhost:9091/healthz
```

### 3. Initialize System

```bash
# Load test documents into vector store
python scripts/init_system.py
```

### 4. Start Backend

```bash
# Run the FastAPI backend
python src/main.py
```

Backend will be available at: http://localhost:8000

### 5. Test System

```bash
# Run test suite
python scripts/test_system.py
```

## Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| API Documentation | http://localhost:8000/docs | - |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| N8N | http://localhost:5678 | admin/admin |
| MinIO Console | http://localhost:9001 | minioadmin/minioadmin |

## Project Structure

```
vinacompare/
├── src/
│   ├── rag/              # RAG pipeline
│   ├── retrieval/        # Search & retrieval
│   ├── models/           # LLM implementations
│   ├── evaluation/       # Evaluation framework
│   ├── monitoring/       # Logging & monitoring
│   └── ui/               # Streamlit interface
├── scripts/              # Utility scripts
├── data/                 # Data storage
├── tests/                # Tests
├── sql/                  # Database schemas
├── prometheus/           # Prometheus config
├── grafana/              # Grafana dashboards
├── docker-compose.yml    # Infrastructure
└── README.md
```

## API Usage

### Query Example (Python)

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "question": "Machine Learning là gì?",
        "model": "Vistral-7B-Chat",
        "top_k": 5,
        "retrieval_mode": "hybrid"
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
```

### Search Documents

```python
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "Docker containerization",
        "top_k": 3,
        "mode": "dense"
    }
)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Search documents
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python programming", "top_k": 3}'

# RAG query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Python là gì?", "top_k": 3}'
```

## Configuration

Edit `.env` file to customize settings:

```env
# Database
POSTGRES_DB=vinarag
POSTGRES_USER=vinarag
POSTGRES_PASSWORD=your_password

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Application
LOG_LEVEL=INFO
ENVIRONMENT=development
```

## Troubleshooting

### Milvus connection failed
```bash
# Check Milvus logs
docker-compose logs milvus-standalone

# Restart Milvus
docker-compose restart milvus-standalone
```

### Model loading OOM
```bash
# Use CPU-only mode
set CUDA_VISIBLE_DEVICES=
python src/main.py
```

### Database connection error
```bash
# Check PostgreSQL
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

## Available Models

- **Vistral-7B-Chat** - Vietnamese Mistral-based model
- **Arcee-VyLinh-3B** - Lightweight Vietnamese model
- **GemSUra-7B** - Google Gemma-based Vietnamese model
- **VinaLLaMA-2.7B** - Vietnamese LLaMA adaptation
- **PhoGPT-7B5** - Vietnamese GPT model

## Next Steps

- Week 3-4: Data ingestion pipeline
- Week 5-6: Multi-model integration
- Week 7: Monitoring & evaluation
- Week 8: MOST submission

## License

MIT License
