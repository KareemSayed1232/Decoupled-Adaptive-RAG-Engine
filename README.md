# Decoupled Adaptive RAG Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
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
  - [1. Clone & Configure](#1-clone--configure)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Build Search Artifacts](#3-build-search-artifacts)
  - [4. Run the Application](#4-run-the-application)
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

-   **Query Expansion with HyDE (Hypothetical Document Embeddings)**
    -   Before retrieval, the engine uses a smaller LLM to generate a hypothetical document that perfectly answers the user's question. Both the original query and this hypothetical document are embedded, significantly improving the semantic richness of the search query and leading to more relevant initial document retrieval.

-   **Hybrid Search (Dense + Sparse Retrieval)**
    -   The system does not rely on a single retrieval method. It combines the strengths of dense, semantic search (using **FAISS**) with traditional sparse, keyword-based search (using **BM25**). The results are merged to ensure that both semantic meaning and specific keywords (like acronyms or names) are captured.

-   **Cross-Encoder Reranking**
    -   The initial set of retrieved documents is passed through a powerful cross-encoder model. Unlike basic vector similarity, a cross-encoder directly compares the query and each document together, providing a much more accurate relevance score. This is a critical step to filter out noise and promote the best possible context for the LLM.

-   **Adaptive Context Strategy (The Core Innovation)**
    -   This is the "adaptive" part of the engine. Instead of naively stuffing all retrieved documents into the prompt, the `ContextBuilder` uses the top reranker score to make an intelligent decision:
        -   **High Confidence:** If the top document's score is above a set threshold, the engine assumes high relevance and provides the LLM with a rich context from multiple top documents.
        -   **Low Confidence (Summarization for Distillation):** If the score is below the threshold, the engine assumes the context might be noisy. It summarizes the top few documents to distill the key facts, providing a concise and factually-grounded context to the LLM while minimizing distraction.
        -   **No Confidence (Rejection):** If the score is critically low, the engine provides no context at all, preventing the LLM from hallucinating based on irrelevant information and allowing it to inform the user that the question is out of scope.

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

-   Python 3.9 or higher
-   Git
-   Access to a terminal or command prompt
---

### 1. Clone & Configure

First, clone the repository and navigate into the project directory.
```bash
git clone https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine.git
cd Decoupled-Adaptive-Rag-Engine
```

Next, create your local environment configuration from the example file.
```bash
cp .env.example .env
```
**Important:** Open the `.env` file and update `LLM_MODEL_PATH` and `HYDE_MODEL_PATH` to point to the correct locations of your GGUF model files.

### 2. Install Dependencies

Dependencies are managed separately for each service.

```bash
# 1. Install the shared data models package
pip install -e ./packages/shared-models

# 2. Install dependencies for each service
pip install -r services/inference-api/requirements.txt
pip install -r services/rag-api/requirements.txt
pip install -r clients/gradio-demo/requirements.txt

### 3. Build Search Artifacts
Run the build script to process your source documents (`data/complete_context.md`) and create the necessary FAISS and BM25 indexes.
python scripts/build_index.py
```
This will populate the `/services/inference_api/artifacts` directory.

### 4. Run the Application

You need to run each of the three services in a **separate terminal**.

| Terminal 1: **Inference API**                  | Terminal 2: **RAG API**                         | Terminal 3: **Gradio UI**                       |
| ---------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| `cd services/inference-api`                    | `cd services/rag-api`                           | `cd clients/gradio-demo`                        |
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

Kareem Sayed - [LinkedIn](https://www.linkedin.com/in/kareem-sayed-dev/) - kareemsayed1232@gmail.com

Project Link: [https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine](https://github.com/KareemSayed1232/Decoupled-Adaptive-Rag-Engine)