# Decoupled Adaptive RAG Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Stars](https://img.shields.io/github/stars/KareemSayed1232/Decoupled-Adaptive-Rag-Engine?style=social)](https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine/stargazers)

A powerful, adaptive Retrieval-Augmented Generation (RAG) system built with a decoupled microservices architecture. This project separates the core AI/ML inference tasks from the business logic, making it scalable, maintainable, and easy to develop.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [RAG Techniques and Innovations](#rag-techniques-and-innovations)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Configure Your Environment](#2-configure-your-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Build Search Artifacts](#4-build-search-artifacts)
  - [5. Run the Application](#5-run-the-application)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Architecture Overview

This project implements a microservices pattern to create a clear separation of concerns. The user interacts with a web client, which communicates with an orchestration API. This API, in turn, offloads all heavy ML computations to a dedicated, independently scalable inference API.

```mermaid
flowchart TD
  %% Define subgraphs and nodes first for clarity
  subgraph UI["User Interface"]
    U(("User"))
    GD(["Gradio Web UI"])
  end

  subgraph RAG["🧠 RAG API @8000"]
    RagApp(["FastAPI Orchestrator"])
    RAG_DB[("Local Artifacts\nchunks.json")]
  end

  subgraph INF["💪 Inference API @8001"]
    IA(["FastAPI ML Endpoints"]) 
    Models(["LLM, Reranker, Embedder"])
    DB[("Vector/Keyword DB\nfaiss.index, bm25.index")]
  end

  U -- Sends Question --> GD
  GD -- HTTP POST /ask --> RagApp
  
  RagApp -- "<b>1. Preprocess Request</b>" --> IA
  RagApp -- "<b>2. Generate HyDE</b>" --> IA
  RagApp -- "<b>3. Embed Texts</b>" --> IA
  RagApp -- "<b>4. Hybrid Search</b>" --> RAG_DB
  RagApp -- "<b>5. Rerank Chunks</b>" --> IA
  RagApp -- "<b>6. Build Context</b>" --> RagApp
  RagApp -- "<b>7. Summarize (if needed)</b>" --> IA
  RagApp -- "<b>8. Generate Stream</b>" --> IA
  
  %% Internal Management in Inference API
  IA -- Manages --> Models
  IA -- Manages --> DB
  
  %% Streaming Data Flow
  U -. Streaming Response .-> GD
  RagApp -. Streaming JSON .-> GD
  IA -. Final Answer Tokens .-> RagApp

  %% Styling
  style RagApp fill:#bbdefb,stroke:#1976d2,stroke-width:2px
  style IA fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
  style UI fill:#fffcf2,stroke:#808080,stroke-width:1px
  style INF fill:#e8f5e9,stroke:#388e3c,stroke-width:1px
	style RAG fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#a185ff
```

## Key Features

-   **Fully Decoupled Services**: Scale, develop, and deploy the UI, logic, and ML services independently.
-   **Real-Time Streaming**: Delivers responses token-by-token for a dynamic and interactive user experience.
-   **Adaptive Context Strategy**: Intelligently builds the final context for the LLM based on retrieval confidence.
-   **Hybrid Search**: Combines dense and sparse retrieval methods for more robust and accurate results.
-   **Clean API Design**: Uses Pydantic for data validation and a shared data models package for type-safe communication.

## RAG Techniques and Innovations

This engine employs a multi-stage RAG pipeline that incorporates several advanced techniques to enhance accuracy and relevance.

-   **Query Expansion with HyDE (Hypothetical Document Embeddings)**: Before retrieval, the engine uses a smaller LLM to generate a hypothetical document that answers the user's question. Embedding both the query and this document significantly improves the semantic richness of the search.
-   **Hybrid Search (Dense + Sparse Retrieval)**: The system combines the strengths of dense, semantic search (FAISS) with traditional sparse, keyword-based search (BM25) to ensure both meaning and specific terms are captured.
-   **Cross-Encoder Reranking**: A powerful cross-encoder model directly compares the query and each retrieved document, providing a much more accurate relevance score to filter out noise and promote the best possible context.
-   **Adaptive Context Strategy**: The engine uses the top reranker score to make an intelligent decision: provide a rich context for high-confidence results, summarize low-confidence results to distill key facts, or reject the context entirely to prevent hallucination.

## Technology Stack

This project is built with a modern stack of technologies, chosen for performance, scalability, and ease of development.

| Category                  | Technology / Library                                                              | Purpose                                                               |
|---------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **Backend & API**         | `FastAPI`, `Uvicorn`                                                              | Building high-performance, asynchronous APIs for both services.       |
| **AI / Machine Learning** | `llama-cpp-python`, `Sentence Transformers`, `mxbai-rerank`, `Transformers`       | Running the LLM, generating embeddings, and reranking documents.      |
| **Vector & Keyword Search** | `Faiss`, `rank_bm25`                                                              | Performing efficient similarity search and keyword-based retrieval.   |
| **Frontend / UI**         | `Gradio`                                                                          | Creating a rapid, interactive web interface for the chat application. |
| **Data & Configuration**  | `Pydantic`, `pydantic-settings`                                                   | Data validation, type safety in APIs, and environment configuration.  |
| **Communication**         | `HTTPX`                                                                           | Asynchronous HTTP client for communication between services.          |

## Getting Started

Follow these steps to get the project up and running on your local machine.

### Prerequisites

-   Python 3.11 or higher
-   Conda
-   Cuda 12.4+
-   Git
-   Access to a terminal or command prompt
-   Visual Studio Build tools


### 1. Clone the Repository
```bash
git clone https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine.git
cd Decoupled-Adaptive-Rag-Engine
```

### 2. Configure Your Environment

#### Create Environment

```bash
conda create -n rag_env python=3.11 -y
conda activate rag_env
```

---

#### 2. Install `llama-cpp-python` with GPU (CUDA 12.4+)

To enable GPU acceleration, you need to compile `llama-cpp-python` from source with CUDA flags.

Run the following inside the environment:

```bash
# set CUDA build flags (new GGML system)
$env:CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on"
$env:FORCE_CMAKE="1"

# install with CUDA support
pip install --force-reinstall --upgrade --no-cache-dir llama-cpp-python --verbose
```

---

#### 3. Verify llama_cpp GPU Support

Run this test:

```python
from llama_cpp import llama_supports_gpu_offload
print("CUDA enabled:", llama_supports_gpu_offload())
```

Expected output:

```
CUDA enabled: True
```

---
#### 4. Install Pytorch with cuda enabled

```python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu[YOUR CUDA VERSION]

# example for cuda 12.4: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

```
#### 5. Verify torch GPU Support

Run this test:

```python
import torch
print("CUDA enabled:", torch.cuda.is_available())
```
Expected output:

```
True
```


All project settings are managed in a single `.env` file. First, create your local copy from the example file:
```bash
Copy-Item .env.example .env
```
Next, open the `.env` file and modify the settings as needed.

#### **Required Settings**
These paths **must be updated** to point to the location of your downloaded GGUF model files on your local machine.

| Variable            | Description                                      | Example Value                               |
|---------------------|--------------------------------------------------|---------------------------------------------|
| `LLM_MODEL_PATH`    | Path to the main Large Language Model file.      | `data/models/guff/Qwen3-8B-Q5_K_M.gguf`       |
| `HYDE_MODEL_PATH`   | Path to the smaller LLM used for HyDE.           | `data/models/guff/Phi-3-mini-4k-instruct-Q4_K_M.gguf` |

These paths **must be updated** to point to the location of custom 'knowledge base' / 'data' on your local machine.

| Variable                  | Description                                                                | Example Value |
|---------------------------|----------------------------------------------------------------------------|---------------------------------------------|
| `BASE_CONTEXT_FILE`       | Path to your base context which is an introduction about the business      | `data/base_context.txt`|
| `COMPLETE_CONTEXT_FILE`   | Path to your complete context and full knowledge about the business        | `data/complete_context.txt` |

#### **Performance & Behavior Tuning (Optional)**
These parameters control the behavior of the RAG pipeline. The default values are a good starting point, but you can tune them for different results.

| Variable                         | Description                                                              | Default |
|----------------------------------|----------------------------------------------------------------------------|---------|
| `RERANKER_REJECTION_THRESHOLD`   | The minimum score from the reranker to consider a document relevant.       | `0.3`   |
| `SUMMARIZATION_MIN_DOCS`         | The number of low-confidence documents to summarize for context.           | `3`     |
| `SS_TOP_K_NEIGHBORS`             | The number of initial documents to retrieve from search.                   | `10`    |
| `GEN_MAX_TOKENS`                 | The maximum number of tokens the LLM can generate in a response.           | `1536`  |
| `SUMMARIZATION_MIN_DOCS`         | The minimum number of chunks needed to be existing before start summarizing| `1536`  |

### 3. Install Dependencies

Dependencies are managed separately for each service.

```bash
# 1. Install the shared data models package
pip install -e ./packages/shared-models

# 2. Install dependencies for each service
pip install -r services/inference_api/requirements.txt
pip install -r services/rag_api/requirements.txt
pip install -r clients/gradio-demo/requirements.txt
```

### 4. Build Search Artifacts

Run the build script to process your source documents (`data/complete_context.md`) and create the necessary FAISS and BM25 indexes.
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python scripts/build_index.py
```
This will populate the `/services/inference_api/artifacts` directory.

### 5. Run the Application

You need to run each of the three services in a **separate terminal**.

| Terminal 1: **Inference API**                  | Terminal 2: **RAG API**                         | Terminal 3: **Gradio UI**                       |
| ---------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| `$env:KMP_DUPLICATE_LIB_OK="TRUE"`             | `$env:KMP_DUPLICATE_LIB_OK="TRUE"`              |                                                 |
| `cd services/inference_api`                    | `cd services/rag_api`                           | `cd clients/gradio-demo`                        |
| `uvicorn src.main:app --port 8001`             | `uvicorn src.main:app --port 8000`              | `python app.py`                                 |

Once all services are running, open your browser and navigate to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

## Project Structure

<details>
<summary>Click to view the detailed project tree</summary>

```
.
├── clients/gradio-demo/      # Frontend UI service
├── data/                     # Source documents and models (not in Git)
├── packages/shared-models/   # Shared Pydantic models for APIs
├── scripts/                  # Helper scripts (e.g., build_index.py)
├── services/
│   ├── inference_api/        # Handles all ML model inference
│   └── rag_api/              # Orchestrates the RAG logic
├── .env.example              # Environment variable template
├── .gitignore                # Specifies files for Git to ignore
├── LICENSE                   # Project license file
└── README.md                 # You are here!
```
</details>

## Contributing

Contributions are welcome and greatly appreciated. Please feel free to fork the project, create a feature branch, and open a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`)
3.  Commit your Changes (`git commit -m 'Add some NewFeature'`)
4.  Push to the Branch (`git push origin feature/NewFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` file for more information.

## Contact

Kareem Sayed - [LinkedIn](https://www.linkedin.com/in/kareem-sayed-dev/) - kareemsaid1232@gmail.com

Project Link: [https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine](https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine)
