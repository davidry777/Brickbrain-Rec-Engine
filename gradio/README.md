# 🧱 Gradio Interface Files

This folder contains all Gradio interface components for the Brickbrain LEGO Recommendation System.

## 📁 Contents

### Main Interface Files
- **`gradio_interface.py`** - Full-featured Gradio interface with all capabilities
- **`gradio_launcher.py`** - Simple demo launcher for quick testing
- **`GRADIO_README.md`** - Comprehensive documentation and setup guide

## 🚀 Quick Start

### Option 1: Simple Demo (Host System)
```bash
# From the project root directory
pip install gradio requests
python3 gradio/gradio_launcher.py
```

### Option 2: Full Interface (Container)
```bash
# From the project root directory
./scripts/launch_gradio.sh
```

### Option 3: Docker Service
```bash
# From the project root directory
docker-compose up -d
```

## 🔧 Testing

Before launching the interface, test the API:
```bash
# From the project root directory
python3 tests/unit/test_gradio_setup.py
```

## 📖 Documentation

For detailed setup instructions, troubleshooting, and feature documentation, see [`GRADIO_README.md`](GRADIO_README.md).

## 🌐 Access Points

Once running:
- **Gradio Interface**: http://localhost:7860
- **API Documentation**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/health

---

**Prerequisites**: Make sure your Docker Compose services are running:
```bash
docker-compose up -d
```
