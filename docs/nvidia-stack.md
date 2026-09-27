# NVIDIA AI Stack — map v1

- [NVIDIA AI Stack — map v1](#nvidia-ai-stack--map-v1)
  - [1. Stack diagram v1](#1-stack-diagram-v1)
  - [2. Summary matrix](#2-summary-matrix)
  - [3. Per-technology cards](#3-per-technology-cards)
    - [3.1 CUDA](#31-cuda)
    - [3.2 CUDA-X libraries](#32-cuda-x-libraries)
    - [3.3 TensorRT](#33-tensorrt)
    - [3.4 TensorRT-LLM](#34-tensorrt-llm)
    - [3.5 Triton Inference Server](#35-triton-inference-server)
    - [3.6 Dynamo](#36-dynamo)
    - [3.7 NIM](#37-nim)
    - [3.8 NeMo](#38-nemo)
    - [3.9 NeMo Agent Toolkit](#39-nemo-agent-toolkit)
    - [3.10 RAPIDS](#310-rapids)
    - [3.11 Morpheus](#311-morpheus)
  - [4. Done when](#4-done-when)
  - [5. Reflection (end of day)](#5-reflection-end-of-day)
  - [Change log](#change-log)

> **Week 1 · Day 2** · Deliverable for `nvidia-ai-plan/week-01/day-02`
> Status: **draft** · Last updated: **2026-09-27** · Revisit: end of Week 8 (v2, after every layer is hands-on)

**Purpose.** Place every framework: what it solves, what it sits on, when to use it, and — most important — **when not to**.
This is the reference I'll use to answer: *"Given a business requirement, choose the appropriate technology and explain why."*

**Rules for filling this in**

- Write in my own words. If a line reads like a product page, rewrite it.
- Every **When-not** must name a concrete alternative or condition (e.g. "< 1 model, low QPS → vLLM").
- Any claim about naming, versions, or hardware support gets a **Source checked** date. Don't inherit claims from `PLAN_REVIEW.md`.
- Skim depth only today. Each tech gets its own week later; mark anything unclear under *Open questions*.

---

## 1. Stack diagram v1

Layers go bottom (hardware) → top (applications). Arrows read **"depends on / runs on"**.
Solid arrow = hard dependency · dashed arrow = optional / pluggable backend · dashed grey box = non-NVIDIA alternative at that layer.

```mermaid
flowchart TB
    subgraph L7["L7 · Applications"]
        APP["FinCrime use cases<br/>fraud scoring · AML triage · investigator agents"]
    end
    subgraph L6["L6 · Frameworks (train / data / agents)"]
        NEMO["NeMo<br/>train · fine-tune · guardrails · eval"]
        NAT["NeMo Agent Toolkit<br/>agent profiling · eval · observability"]
        RAPIDS["RAPIDS<br/>cuDF · cuML · cuGraph"]
        MORPHEUS["Morpheus<br/>streaming security / fraud pipelines"]
        ALT6["ADK · LangGraph · HF TRL/PEFT<br/>pandas · Polars · Flink"]:::alt
    end
    subgraph L5["L5 · Packaged microservices"]
        NIM["NIM<br/>OpenAI-compatible model container"]
        ALT5["Ollama · hosted APIs<br/>Bedrock · Vertex · Azure OpenAI"]:::alt
    end
    subgraph L4["L4 · Serving / orchestration"]
        TRITON["Triton Inference Server<br/>multi-model · multi-backend"]
        DYNAMO["Dynamo<br/>distributed LLM serving<br/>disagg. prefill/decode · KV-aware routing"]
        ALT4["TGI · KServe · Ray Serve"]:::alt
    end
    subgraph L3["L3 · Engines / optimization / compilation"]
        TRT["TensorRT<br/>AOT inference compiler"]
        TRTLLM["TensorRT-LLM<br/>LLM engine + runtime"]
        PT["PyTorch + Megatron-Core"]:::alt
        ALT3["vLLM · SGLang · llama.cpp"]:::alt
    end
    subgraph L2["L2 · Libraries (CUDA-X)"]
        CUDAX["cuBLAS · cuDNN · NCCL · CUTLASS · …"]
    end
    subgraph L1["L1 · Programming model"]
        CUDA["CUDA toolkit + runtime"]
    end
    subgraph L0["L0 · Hardware"]
        HW["NVIDIA GPU + driver<br/>Ampere · Ada · Hopper · Blackwell"]
    end

    %% L1-L2: foundation
    CUDA --> HW
    CUDAX --> CUDA

    %% L3: engines
    TRT --> CUDAX
    TRTLLM --> TRT
    TRTLLM --> CUDAX
    PT --> CUDAX
    ALT3 --> CUDAX

    %% L4: serving
    TRITON -->|backend| TRT
    TRITON -->|backend| TRTLLM
    TRITON -.->|backend| PT
    TRITON -.->|vLLM backend| ALT3
    DYNAMO -->|engine| TRTLLM
    DYNAMO -.->|engine| ALT3
    ALT4 --> ALT3

    %% L5: packaging
    NIM -->|engine inside| TRTLLM
    NIM -.->|engine inside| ALT3
    NIM -.-> TRITON

    %% L6: frameworks
    NEMO --> PT
    NEMO -.->|export / deploy| TRTLLM
    NEMO -.->|deploy| NIM
    NAT -.->|any OpenAI-compatible LLM| NIM
    RAPIDS --> CUDAX
    MORPHEUS --> RAPIDS
    MORPHEUS --> TRITON

    %% L7: applications
    APP --> MORPHEUS
    APP --> RAPIDS
    APP --> NAT
    APP --> NIM

    classDef alt fill:#f4f4f4,stroke:#999,stroke-dasharray:5 4,color:#555
```

**Diagram notes** (anything the boxes can't show — e.g. things that sit on *two* layers, optional dependencies):

- **Edge direction** — every arrow points from the thing to what it needs. Following arrows down from any box gives its full dependency chain (e.g. Morpheus → Triton → TensorRT → CUDA-X → CUDA → GPU).
- **TensorRT-LLM straddles L3/L4** — it's an engine, but it also ships its own runtime and an OpenAI-compatible server (`trtllm-serve`), so it can serve on its own without Triton or Dynamo.
- **Triton vs Dynamo sit side by side, not stacked** — Triton serves *any* model type (TensorRT, ONNX, PyTorch, FIL/XGBoost, Python); Dynamo is LLM-only and orchestrates engines (TRT-LLM, vLLM, SGLang) across nodes. Dynamo does **not** require Triton.
- **Dynamo is the only NVIDIA box that points at non-NVIDIA engines as first-class** — via vLLM/SGLang it also runs on AMD / Intel hardware, which is why its "depends on" is really "an engine", not "a GPU".
- **NIM is packaging, not a new engine** — inside a NIM container is one of TRT-LLM or vLLM (chosen per model/GPU profile). The dashed edge to Triton marks that some NIMs are built on it.
- **NeMo spans two layers** — training/fine-tuning sits on PyTorch + Megatron-Core (L3), while its outputs are deployed through TRT-LLM / NIM. NeMo Guardrails can sit in front of *any* endpoint at runtime.
- **NeMo Agent Toolkit needs no GPU** — it only needs an LLM endpoint (NIM is optional; OpenAI, Bedrock, etc. work too). This is where my Google ADK platform plugs in (`nat.plugins.adk`).
- **RAPIDS is technically part of CUDA-X** — drawn at L6 because I use it as a framework, but it skips L3–L5 entirely and talks straight to CUDA-X.
- **Morpheus is a composer** — it has no layer of its own below; it stitches RAPIDS (feature engineering) + Triton (inference) into a streaming pipeline, fed by Kafka (not drawn).
- **Grey dashed boxes = the when-not escape hatch** for each layer — the non-NVIDIA option I'd pick when the NVIDIA box is overkill.

---

## 2. Summary matrix

One line per cell. Detail goes in §3.


| Tech               | What it solves                                                                                                                                           | Depends on                                                                       | Use when                                                                                                                                               | **Don't use when**                                                                                                               | Non-NVIDIA alternative                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| CUDA               | programming model to communicate with Nvidia GPU                                                                                                         | Nvidia GPU and Drivers                                                           | LLM training and inference                                                                                                                             | Sequential, single-threaded operations or non-NVIDIA deployments                                                                 | ROCm/HIP, Metal, Triton-lang, WebGPU                                                                                |
| CUDA-X             | suite of libraries designed to deliver industry-leading GPU acceleration across AI and high-performance computing use cases                              | Nvidia GPU, CUDA Toolkit and Drivers                                             | Building high-throughput, low-latency AI and data processing pipelines over massive datasets                                                           | Handling lightweight, low-volume datasets that easily fit in single-core CPU memory, or running on non-NVIDIA hardware platforms | NetworkX, scikit-learn, NumPy, Apache Spark, Ray and PyTorch                                                        |
| TensorRT           | ecosystem of tools for developers to achieve high-performance deep learning inference                                                                    | `libcudart`, `cuDNN`, GPU/Drivers                                                | Deploying latency-critical, high-throughput deep learning inference                                                                                    | Running inference on non-Nvidia hardware, Rapid iterative prototyping, Serving pure tabular ML models                            | ONNX Runtime, torch.compile, vLLM/SGLang, OpenVINO                                                                  |
| TensorRT-LLM       | open-source library specifically built to compile, optimize, and serve Generative AI and Large Language Models (LLMs) on NVIDIA GPUs.                    | TensorRT, CUDA Driver/Toolkit, Python, `cuDNN`, `NCCL`, `MPI`                    | Deploying large-scale Transformer models to production on Nvidia hardware                                                                              | Deploying non-generative models, or working with non-Nvidia hardware, building quick Python-only prototypes                      | vLLM, SGLang, TGI, LMDeploy, Ollama                                                                                 |
| Triton             | open source inference serving software that streamlines AI inferencing                                                                                   | GPU/Drivers/CUDA, Python/C++, model artifacts and `config.pbtxt`                 | High-throughput & low-latency requirements, Heterogeneous multi-framework stacks, Complex model pipelines                                              | Simple lightweight deployments, Python-native rapid prototyping, Strict CPU-only microservices                                   | KServe, TorchServe, Ray Serve, BentoML                                                                              |
| Dynamo             | open-source inference framework for serving generative AI models                                                                                         | engine backend (vLLM/SGLang/TensorRT-LLM), container platform (Kubernetes/Slurm) | Serving scalable LLMs with KV-cache-aware routing                                                                                                      | Single-node/single-GPU setups                                                                                                    | KServe, Ray Serve, Triton, TGI, DynamoDB                                                                            |
| NIM                | set of containerized microservices that facilitates the deployment and inference of pretrained models                                                    | GPU/Drivers/CUDA, Docker/Kubernetes, TensorRT/TensorRT-LLM, vLLM/Triton          | Self-host/control AI inference infrastructure, Out-of-the-box optimization, Standard OpenAI-compatible endpoints                                       | Using hosted SaaS LLM APIs, Non-Nvidia Hardware, Constrained small-scale non-commercial projects                                 | Open/Local: vLLM, Ollama, TGI, Triton, LocalAIManaged: AWS Bedrock, Vertex AI, Azure OpenAI, Anyscale, Together AI |
| NeMo               | a modular software suite of APIs and libraries that help developers manage the AI agent lifecycle—building, deploying, and optimizing AI agents at scale | GPU/Drivers/CUDA, PyTorch, Megatron-LM Triton/NIM, Kubernetes/NVAIE              | Building large foundation models, High-throughput inference and low-latency safety controls, Integrated enterprise stack with commercial SLA via NVAIE | Single GPU and local workstation prototyping, Non-Nvidia hardware, Simple API Aggregation                                        | Hugging Face, DeepSpeed, LLama-Factory, LLama Guard, Guidance, Ray Data, Datatrove                                  |
| NeMo Agent Toolkit | flexible, lightweight, and unifying library that allows you to easily connect existing enterprise agents to data sources and tools across any framework  | Python, Pydantic, Dask, FastAPI, MCP, A2A, OpenTelemtry/OpenInference, NIMs/LLMs | Single unified gateway for different agents frameworks, Standard enterprise middleware features, Capture step-by-step agent execution trajectories     | Standalone single-framework agent, Basic prompt tracing, Standard LLM observability                                              | LangSmith, LangFuse, LangGraph Server, Semantic Kernel                                                              |
| RAPIDS             | open-source suite of GPU-accelerated software libraries created by NVIDIA to execute end-to-end data science                                             | GPU/Drivers/CUDA, Python, C++, Arrow/PyArrow                                     | Processing massive datasets, Requiring speedups for tabular transofrmations, Classical ML, Graph Operations                                            | Datasets under 100k-1M, CPU-native frameworks like pandas, Polars, DuckDB                                                        | pandas, Polars, DuckDB, Spark, scikit-learn, NetworkX                                                               |
| Morpheus           | GPU-accelerated AI application framework designed for streaming cybersecurity and telemetry data processing                                              | RAPIDS (cuDf, cuML), Triton, Kafka/Flink, C++/Python                             | Ultra-High Throughput Telemetry:, GPU-Bound Analytics & Real-Time Inference, Digital Fingerprinting                                                    | Simple Rule-Based ETL, Massive Distributed Graph Algorithms, Non-GPU Environments                                                | Kafka/Flink, Ray Streaming, Databricks/Spark Streaming                                                              |


---

## 3. Per-technology cards

### 3.1 CUDA

**Docs:** [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/) · **Source checked:** **2026-08-22**, version **13.4.2**

- **What:** CUDA is a parallel computing platform and programming model developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU. It allows developers to accelerate compute-intensive applications and is widely used in fields such as deep learning, scientific computing, and high-performance computing (HPC).
- **Depends on:** Nvidia Driver and a compatible GPU containing execution units like standard CUDA Cores or hardware-accelerated Tensor Cores
- **When:** needing to process large blocks of data such as LLM training and inference. Furthermore, it can be used for matrix multiplication, 3D rendering pipelines, complex physics/fluid simulations, cryptographic processing, and running optimized native C++/Python kernels via tools like `nvcc` or `cuTile`.
- **When-not:** need to run on AMD, Intel or Apple Silicon hardware without a translation layer. Avoid for tasks that are serial, IO bound (web servers), automation scripts where the latency of transferring data over the PCIe bus to GPU memory (`cudaMalloc`) outweighs the parallel processing speedup.
- **Alternative(s):**
  - **Hardware-Specific:** AMD ROCm/HIP (for AMD GPUs), Apple Metal / Metal Performance Shaders (for Macs), Intel oneAPI / SYCL.
  - **Cross-Platform APIs:** OpenCL, Vulkan Compute, WebGPU (for web browsers).
  - **High-Level Languages/Compilers:** OpenAI Triton-lang (Python-like DSL for writing fast GPU kernels that compile to both CUDA and AMD backends).
- **FinCrime angle:** High-throughput algorithmic transaction monitoring, cryptographic hashing, and rapid laundering pattern recognition. In financial crime prevention, CUDA accelerates heavy anti-money laundering (AML) operations by powering parallelized graph neural networks (GNNs) that map complex transaction webs in real-time. It is also the underlying platform used to accelerate fraud detection machine learning models, analyze high-frequency trading anomalies, and computationally drive crypto-asset transaction tracing.
- **Lab week:** `01-cuda`

### 3.2 CUDA-X libraries

**Docs:** [developer.nvidia.com/gpu-accelerated-libraries](https://developer.nvidia.com/gpu-accelerated-libraries) · **Source checked:** **2026-08-22**

- **What:** powerful suite of libraries designed to deliver industry-leading GPU acceleration across AI and high-performance computing use cases including generative AI and autonomous machines to climate modeling and financial forecasting. It also provide highly optimized implementations of complex algorithms that far outperform CPU-only alternatives.
- **Depends on:** NVIDIA GPU hardware, the base CUDA Driver/Toolkit, C++ or Python runtimes
- **Which libraries matter for this plan:**
  - **cuDNN** (CUDA Deep Neural Network library): Highly tuned implementations for foundational deep learning routines like convolutions, pooling, and activation layers.
  - **cuBLAS / CUTLASS** Libraries engineered for high-performance basic linear algebra and matrix multiplications on Tensor Cores.
  - **NCCL** (NVIDIA Collective Communications Library): Multi-GPU and multi-node collective communication primitives utilized during heavy distributed training.
  - **DALI** (Data Loading and Augmentation Library): A data pipeline acceleration engine designed to optimize data pre-processing for ML mode
  - **TensorRT/LLM** High performance deep learning inference optimization and runtime for production deployment.
  - **NeMo Curator** Improves gen AI model accuracy with pre-build pipelines for generating synthetic data.
  - **Morpheus** Optimizing AI-driven real-time cypersecurity and network streaming pipelines
- **When:** Building high-throughput, low-latency AI and data processing pipelines over massive datasets (e.g., millions/billions of nodes, real-time transaction streams) where CPU processing creates severe throughput bottlenecks.
- **When-not:** Handling lightweight, low-volume datasets that easily fit in single-core CPU memory, or running on non-NVIDIA hardware platforms (such as CPU-only environments or non-CUDA hardware targets).
- **Alternative(s):**
  - CPU-based libraries (NetworkX, scikit-learn, NumPy, Apache Spark).
  - Open-source multi-vendor GPU libraries (ROCm/HIP for AMD, OneAPI/oneDAL for Intel).
  - High-level distributed systems like Ray and PyTorch Distributed.
- **FinCrime angle:** transaction networks are modeled as massive graphs. CUDA-X enables real-time Graph Neural Networks (GNNs) and vector search to detect intricate money laundering rings, mule networks, and anomalous transaction flows across billions of historical entities in milliseconds.
- **Lab week:** *(implicit everywhere)*

### 3.3 TensorRT

**Docs:** [docs.nvidia.com/deeplearning/tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/) · **Source checked:** **2026-09-22**, version **11.3.0**

- **What:** an ecosystem of tools for developers to achieve high-performance deep learning inference. TensorRT includes inference compilers, runtimes, and model optimizations that deliver low latency and high throughput for production applications
- **Depends on:** CUDA toolkit (specifically `libcudart`), `cuDNN` library, Nvidia GPU/Drivers
- **Inputs:**
  - Trained model representations (such as ONNX models or PyTorch exports via Torch-TensorRT)
  - Target precision configuration (FP32, FP16, INT8, FP8, NVFP4)
  - Target GPU hardware profiles
  - Calibration datasets (for quantization).
  - At runtime, it receives live input tensors directly in GPU memory buffers (`void*`).
- **Outputs:**
  - Platform-specific optimized plan/engine files (`.plan` or `.engine`) (binary executables containing optimized GPU execution kernels, fused layers and allocated memory structures).
  - At runtime, outputs prediction tensors (logits, embeddings, or bounding boxes) directly into GPU memory buffers.
- **When:** Deploying latency-critical, high-throughput deep learning inference workloads to production specifically on NVIDIA hardware across data centers, RTX PCs, automotive platforms, or edge devices.
- **When-not:**
  - Running inference on non-NVIDIA hardware targets (such as AMD, Intel, or CPUs).
  - Rapid iterative prototyping where compilation overhead delays real-time debugging.
  - Serving pure tabular or classical machine learning models (like Random Forests or XGBoost) without neural network structures.
- **Alternative(s):**
  - ONNX Runtime: Multi-platform execution engine supporting CPUs, AMD GPUs, and Apple Silicon, offering high cross-hardware portability.
  - torch.compile: Native PyTorch JIT graph compiler for fast Pythonic model optimization without exporting models.
  - vLLM / SGLang: Specialized LLM serving engines optimized for high-throughput continuous batching and KV-cache management.
  - OpenVINO: Intel’s dedicated framework for optimizing deep learning models across Intel CPUs, GPUs, and NPUs.
- **FinCrime angle:** in Financial Crime workflows, real-time AI models like deep learning transaction classifiers, GNN fraud scoring engines, and NLP sanctions screening need sub-millisecond latencies to approve or block transactions mid-flight. TensorRT minimizes execution latency and maximizes batch throughput, allowing banks to run complex neural networks within strict SLA limits.
- **Lab week:** `02-tensorrt`

### 3.4 TensorRT-LLM

**Docs:** [nvidia.github.io/TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) · **Source checked:** **2026-09-22**, version **1.3.0rc27**

- **What:** open-source library that accelerates and optimizes inference performance of large language models (LLMs) on the NVIDIA AI platform with a simplified Python API.
- **Depends on:** TensorRT, CUDA Driver/Toolkit, cuDNN, Python runtime, and communication libraries like NCCL or MPI for multi-GPU/multi-node tensor parallelism.
- **What it adds over plain TensorRT:**
  - Out-of-the-box LLM architecture support (Transformer blocks, RoPE)
  - High-level KV-cache management
  - Paged Attention
  - Continuous/in-flight batching
  - Tensor/pipeline parallelism primitives
  - Advanced LLM quantization schemes (INT4 AWQ, FP8, NVFP4)
  - Production C++ runtime ready for NVIDIA Triton Inference Server.
- **When:** Deploying large-scale Transformer models (e.g., Llama 3, Mistral, Qwen, DeepSeek) to production environments on NVIDIA hardware requiring maximum generation throughput and sub-millisecond per-token latency.
- **When-not:**
  - Deploying standard non-generative models (e.g., ResNet, XGBoost, basic CNNs/RNNs—use plain TensorRT or Triton instead).
  - Hosting models on non-NVIDIA GPUs or CPU-only infrastructure.
  - Building quick Python-only prototypes where setting up C++ engines and Cuda graph compilations adds unnecessary friction.
- **Alternative(s):** vLLM, SGLang, Hugging Face TGI (Text Generation Inference), LMDeploy, Ollama (for local dev).
- **Effort vs. payoff vs. vLLM (gut call, verify in Wk 4):**
  - vLLM: Low-to-medium effort, extremely fast setup, excellent baseline performance.
  - TensorRT-LLM: High deployment/build effort (engine compilation, schema alignment, driver tuning), but yields maximum hardware efficiency and peak token throughput at scale.
- **FinCrime angle:** Financial crime units deploy LLMs for automated suspicious activity report (SAR) generation, real-time sanctions screening narrative analysis, and deep document processing (KYC). TensorRT-LLM ensures these high-parameter LLMs run fast enough to analyze massive transaction streams without breaching compliance processing SLAs.
- **Lab week:** `03-tensorrt-llm`

### 3.5 Triton Inference Server

**Docs:** [github.com/triton-inference-server/server](https://github.com/triton-inference-server/server) · **Source checked:** **2026-09-22**, version **2.72.0**

- **What:** open source inference serving software that enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more.
- **Depends on:** GPU/Drivers/CUDA, Python/C++, file-system-based storage repositories (S3, GCS, Azure Blob, local directory) containing model artifacts and `config.pbtxt`
- **Backends it can host:**
  - Deep Learning: TensorRT, TensorRT-LLM, PyTorch, ONNX Runtime, OpenVINO, vLLM.
  - ML/Tabular: RAPIDS FIL (Forest Inference Library for XGBoost, LightGBM, Random Forest).
  - General Purpose / Orchestration: Python Backend (custom Python code), C++ / Custom backends, DALI (GPU-accelerated data pre-processing).
- **When:**
  - Maximum hardware efficiency via dynamic batching, concurrent model execution and GPU memory sharing.
  - Serving models built across different framework
  - Native ensembling or Business Logic Scripting (BLS) to route data between pre-processors, embedding generators and ML classifiers directly on the server without extra network hops.
- **When-not:**
  - Single-framework, microservice-style APIs where Triton’s complex model configuration files `config.pbtxt` and infrastructure overhead are overkill.
  - Apps with lightweight Python wrappers (like FastAPI) and don't require GPU batching or multi-model orchestration.
- **Alternative(s):**
  - KServe: Higher-level Kubernetes orchestration platform for model serving (note: KServe can use Triton as its underlying runtime container).
  - TorchServe: PyTorch-native model server designed specifically for PyTorch/TorchScript models.
  - Ray Serve: Python-first scalable model serving framework best suited for distributed Python execution and dynamic scaling across clusters.
  - BentoML: Python-first deployment tool focusing on packaging, containerizing, and orchestrating ML models with simple REST/gRPC endpoints.
- **FinCrime angle:**
  - Single Endpoint Pipeline: Triton is ideal for Financial Crime (FinCrime) architectures that process high-frequency transaction data. You can expose a single gRPC/REST endpoint that triggers an Ensemble or Business Logic Scripting (BLS) pipeline.
  - Concrete Flow:
    - Pre-processing: Clean/format transaction attributes (via DALI or Python backend).
    - Embedding Generation: Pass text/transaction fields to a deep learning model (PyTorch / ONNX) to generate transaction embeddings.
    - Fraud Scoring: Pass those embeddings alongside tabular features into an XGBoost model via the RAPIDS FIL backend.
    - Output: Return a single unified risk score and explanation vector to the upstream payment gateway in a single low-latency round trip.
- **Lab week:** `04-triton-dynamo`

### 3.6 Dynamo

**Docs:** [docs.nvidia.com/dynamo](https://docs.nvidia.com/dynamo) · **Source checked:** **2026-09-22**, version 

- **What:** an open-source inference framework for serving generative AI models. It supports TensorRT-LLM, vLLM, SGLang. It also supports Nvidia, AMD and Intel hardware, can be run on Kubernetes, Slurm or locally.
- **Depends on:** inference engine backend (vLLM, SGLang, or TensorRT-LLM) to perform the actual model execution, and container orchestration platforms like Kubernetes or Slurm.
- **Scale at which it starts to matter:** High-concurrency or enterprise-scale AI serving clusters (typically multi-node deployments running high QPS workloads, massive context windows, or split prefill/decode disaggregated topologies where standard monolithic LLM serving becomes inefficient).
- **When:** serving LLMs at scale in production and need KV-cache-aware intelligent request routing, disaggregated prefill/decode serving, cross-node cache management, and granular auto-scaling across heterogeneous clusters.
- **When-not:** Single-node/single-GPU setups, small-scale deployments, low-throughput applications, or simple prototype applications where standard vLLM/SGLang Docker containers or lightweight API servers (like Ollama or LM Studio) are sufficient and introduce less operational complexity.
- **Alternative(s):**
  - Ray Serve (with vLLM/SGLang)
  - Triton Inference Server
  - TGI (Text Generation Inference)
  - KServe
  - DynamoDB (if referring to the unrelated AWS key-value store database).
- **Hardware support note:** Optimized for NVIDIA GPUs, but provides multi-vendor hardware support including AMD GPUs (ROCm) and Intel XPUs.
- **FinCrime angle:** Serves as high-throughput, low-latency infrastructure for running real-time LLM-based financial crime monitoring pipelines—such as real-time transaction screening, automated SAR (Suspicious Activity Report) generation, AML multi-agent investigations, and fraud detection at scale with predictable SLAs.
- **Lab week:** `04-triton-dynamo`

### 3.7 NIM

**Docs:** [docs.nvidia.com/nim](https://docs.nvidia.com/nim/) · **Source checked:** **2026-09-22**

- **What:** a set of pre-packaged, containerized microservices designed to simplify and accelerate the production deployment of generative AI models (LLMs, vision, speech, biology/healthcare, etc.) on accelerated infrastructure.
- **Depends on:** Container runtimes (Docker, Kubernetes/Helm), NVIDIA CUDA libraries, TensorRT / TensorRT-LLM, vLLM / Triton Inference Server, and host environments with compatible NVIDIA GPUs (cloud, on-prem DGX, or RTX workstations).
- **API surface:** Standardized OpenAI-compatible HTTP/REST and gRPC APIs (e.g., `/v1/chat/completions`, `/v1/embeddings`), allowing developers to swap underlying backends with minimal code changes.
- **Licensing / deployment model:**
  - Prototyping / Development: Free hosted endpoints via the NVIDIA API catalog or local development using NVIDIA Developer Program credits.
  - Production: Part of the commercial NVIDIA AI Enterprise subscription license (priced per GPU/socket/node), required for self-hosting in production environments.
  - Deployment Options: Fully portable across public clouds (AWS, GCP, Azure), on-prem data centers, Kubernetes clusters, DGX Cloud, and local RTX PCs.
- **When:**
  - You need to self-host or control AI inference infrastructure on-premises or in private clouds for data privacy, compliance, or low-latency SLAs.
  - You want out-of-the-box optimization (throughput, latency, memory usage) for open/custom models without manually tuning Triton, TensorRT, or CUDA dependencies.
  - You want standard OpenAI-compatible endpoints to seamlessly integrate with existing orchestration tools (LangChain, LlamaIndex, enterprise microservices).
- **When-not:**
  - You prefer using hosted SaaS LLM APIs (e.g., OpenAI, Anthropic, Google Gemini) without managing any infrastructure.
  - You do not run on NVIDIA GPU hardware (e.g., CPU-only, Apple Silicon, AWS Trainium/Inferentia, Google TPUs).
  - You are running constrained, small-scale non-commercial projects where licensing NVIDIA AI Enterprise is cost-prohibitive.
- **Alternative(s):**
  - Open-source / Self-hosted engine stacks: vLLM, Ollama, TGI (Text Generation Inference), Triton Inference Server, LocalAI.
  - Managed Cloud Services: AWS Bedrock, Google Cloud Vertex AI, Azure OpenAI Service, Anyscale, Together AI.
- **FinCrime angle:**
  - On-Premises Privacy & Data Sovereignty: Financial institutions handling sensitive PII, transactional data, and AML (Anti-Money Laundering) checks often cannot send data to third-party SaaS APIs. NIM allows running performant LLMs completely air-gapped or within private VPCs.
  - Low Latency for Real-time Fraud Detection: NIM's optimized inference runtime delivers sub-second response times necessary for processing real-time credit card transactions, sanction screening, and transaction monitoring pipelines.
  - Auditability & Security: Packaged with enterprise-grade hardened base containers and regular CVE security patching, meeting strict cybersecurity and regulatory standards in banking.
- **Lab week:** `05-nim`

### 3.8 NeMo

**Docs:** [docs.nvidia.com/nemo](https://docs.nvidia.com/nemo/) · **Source checked:** **2026-09-22**, version **26.02**

- **What:** is a modular, enterprise-grade software platform and open-source framework designed for the end-to-end development, customization, guardrailing, and deployment of generative AI models (LLMs, vision-language models, and speech/audio).
- **Depends on:**
  - Hardware: NVIDIA GPUs (Ampere, Hopper, Blackwell architectures) with CUDA ecosystem support.
  - Core Libraries & Orchestration: PyTorch, PyTorch Lightning, Megatron-LM (for distributed multi-GPU/multi-node training), and Triton Inference Server / NVIDIA NIM (for production serving).
  - Enterprise Requirements: Kubernetes / NVIDIA AI Enterprise (NVAIE) license for full containerized NeMo Microservices features.
- **Sub-components I need to know:**
  - NeMo Framework: The underlying open-source Python framework for pre-training and fine-tuning multi-billion parameter models across distributed GPU nodes.
  - NeMo Curator: GPU-accelerated microservice/library for raw data extraction, cleaning, deduplication, and quality filtering.
  - NeMo Guardrails: Programmable safety layer using `Colang` to filter prompt injections, hallucinations, toxic content, and maintain topical boundaries.
  - NeMo Retriever: Accelerated microservices for Enterprise RAG (document ingestion, parsing, chunking, and high-throughput vector embedding search).
  - NeMo Customizer: Microservice specializing in parameter-efficient fine-tuning (`PEFT`/`LoRA`), `SFT`, and alignment (`RLHF`/`DPO`).
  - NeMo Evaluator: Automated benchmark suite (`MMLU`, `GSM8K`, `LLM-as-a-judge`, custom domain metrics) to validate models post-fine-tuning.
- **When:**
  - You are building or customizing large foundation models on multi-GPU or multi-node infrastructure.
  - High-throughput inference and low-latency safety controls are critical requirements.
  - You need an integrated enterprise stack (data preparation -> fine-tuning -> evaluation -> guardrails) with commercial SLA/support via NVIDIA AI Enterprise.
- **When-not:**
  - Single-GPU / Local Workstation prototyping: Lightweight OSS stacks are faster to setup and carry significantly lower overhead.
  - Non-NVIDIA Hardware: It is heavily optimized for CUDA; running on AMD (ROCm), Apple Silicon, or TPUs is impractical or unsupported.
  - Simple API Aggregation: If you only need to call external APIs (e.g., OpenAI, Anthropic) without hosting or fine-tuning models yourself.
- **Alternative(s):**
  - Training & Fine-tuning: Hugging Face (TRL/PEFT), Axolotl, Unsloth, DeepSpeed, LLaMA-Factory.
  - Guardrails & Safety: Guardrails AI, Llama Guard, Guidance (Microsoft), NeMo Guardrails (can be used standalone).
  - Data Processing: Ray Data, Datatrove (Hugging Face).
- **FinCrime angle:**
  - Guardrails: Prevents model manipulation or prompt injections trying to bypass anti-money laundering (AML) controls or extract restricted compliance logic.
  - PII & Data Leakage: NeMo Safe Synthesizer and Curator mask/replace sensitive tabular customer data (SSNs, account numbers) and control model context output to satisfy privacy compliance.
  - Model Risk / SR 11-7: NeMo Evaluator and NeMo Auditor enable reproducible benchmarks, bias auditing, and model tracking. This provides the audit trails, validation metrics, and governance docs required by OCC/Federal Reserve SR 11-7 model risk management guidance.
- **Lab week:** `06-nemo-finetuning`

### 3.9 NeMo Agent Toolkit

**Docs:** [docs.nvidia.com/nemo/agent-toolkit](https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html) · **Source checked:** **2026-09-22**, version **1.8**

- **What:** a flexible, lightweight, and unifying library designed to connect, coordinate, observe, and optimize enterprise AI agents, tools, and data sources across different agentic frameworks.
- **Depends on:**
  - Core: Python runtime, Pydantic, Dask (for async job execution), FastAPI (for server/REST endpoints).
  - Protocols & Integrations: Model Context Protocol (MCP), Agent-to-Agent protocol (A2A), OpenTelemetry / OpenInference for tracing.
  - LLMs/Hardware: Works with NVIDIA NIMs/local LLMs as well as third-party providers (OpenAI, AWS Bedrock, OCI).
- **Is it an agent framework or a layer around one?** It is primarily a layer around agent frameworks. While it includes built-in reference agent implementations (such as ReAct, ReWOO, and Router agents), its core value is acting as an orchestration, integration, and middleware layer that wraps, connects, and standardizes workflows built on external frameworks (e.g., LangGraph, AutoGen, Agno/Phidata, Google ADK).
- **How it relates to my Google ADK platform:** NeMo Agent Toolkit provides an explicit ADK integration plugin (`nat.plugins.adk`).
  - Wrapper/Bridge: It wraps Google ADK agents and tools, translating native ADK execution calls and tool wrappers into NAT's standardized schema.
  - Standardized Infrastructure: Allows you to run ADK-built agents through NAT's FastAPI frontend, expose them via MCP/A2A servers, and apply NAT's middleware (caching, timeout handling, authentication/identity resolution) without rewriting your core ADK agent logic.
  - Unified Observability & Optimization: Connects Google ADK agents directly into NAT’s unified telemetry pipeline (OpenTelemetry), dataset generators, and fine-tuning harness (`nat.finetuning`).
- **When:**
  - You already have agents written in different frameworks (e.g., Google ADK, LangGraph, AutoGen) and need a single, unified gateway or orchestration layer to expose them via standardized APIs (REST, MCP, A2A).
  - You need standard enterprise middleware features (user identity resolution, OAuth2/token storage, rate-limiting, dynamic tool registration) across all agent workflows.
  - You want to capture step-by-step agent execution trajectories to evaluate, profile, or fine-tune lower-level models (using NAT’s Test-Time Compute or DPO fine-tuning harnesses).
- **When-not:**
  - You are building a standalone, single-framework agent entirely inside Google ADK or LangGraph and do not need multi-framework interoperability or standardized middleware servers.
  - You only need basic prompt tracing or standard LLM observability without relying on NeMo's trajectory-based evaluation and fine-tuning pipelines.
- **Alternative(s):**
  - Observability & Trajectory Tracing: LangSmith, Langfuse, Arize Phoenix, Weights & Biases Weave.
  - Agent Interoperability & Gateway Wrappers: AutoGen Studio, LangGraph Server, Semantic Kernel.
- **FinCrime angle:**
  - Comprehensive Execution Trajectories: Captures structured step-by-step trajectories (including user context, planner reasoning, tool inputs, API paylod responses, and final actions) via standardized OpenTelemetry spans.
  - Immutable Audit Trail: Provides the detailed provenance required by financial compliance teams to explain why an agent made a decision (e.g., flagging a transaction, filing a SAR, or approving an override).
  - Contextual Redaction: Features built-in PII/PHI redaction processors (`nat.observability.processor.redaction`) to scrub sensitive financial data, SSNs, or account numbers from execution logs before they are written to long-term storage or telemetry pipelines.
- **Lab week:** `07-nemo-agent-toolkit`

### 3.10 RAPIDS

**Docs:** [rapids.ai](https://rapids.ai/) · **Source checked:** **2026-09-22**

- **What:** open-source suite of GPU-accelerated software libraries created by NVIDIA to execute end-to-end data science, analytics, and machine learning pipelines entirely on GPUs using familiar Python APIs.
- **Depends on:** NVIDIA GPUs (Pascal architecture or newer), CUDA, C++, and Arrow/PyArrow memory layouts.
- **cuDF / cuML / cuGraph — one line each:**
  - cuDF: A GPU-accelerated DataFrame library featuring zero-code-change drop-in acceleration for pandas and integration with `Polars` and `Spark`.
  - cuML: A library providing GPU-accelerated implementations of classical machine learning algorithms and estimators mirroring the `scikit-learn` API.
  - cuGraph: A high-performance graph analytics library that integrates seamlessly with `NetworkX` to process large-scale graph structures on GPUs.
- **When:** processing massive datasets (millions to billions of rows) that bottleneck CPU memory/compute, requiring speedups for tabular transformations, classical ML, or graph operations.
- **When-not:** Datasets under ~100k–1 million rows or a few hundred megabytes, where GPU memory transfer overhead outweighs compute acceleration and CPU-native frameworks like pandas, Polars, or DuckDB run faster.
- **Alternative(s):** `pandas`, `Polars`, `DuckDB`, `Spark`, `scikit-learn`, `NetworkX`
- **FinCrime angle:** Rapidly constructs massive transaction graphs across millions of accounts to detect money laundering rings, shell company structures, and fraud schemes via parallelized community detection and centrality algorithms (`cuGraph`), paired with fast fuzzy string matching and behavioral feature extraction for entity resolution (`cuDF`).
- **Lab week:** `08-rapids`

### 3.11 Morpheus

**Docs:** [docs.nvidia.com/morpheus](https://docs.nvidia.com/morpheus/) · **Source checked:** **2026-09-22**, version **25.06**

- **What:** open, GPU-accelerated AI application framework designed for streaming cybersecurity and telemetry data processing. Built around C++ and Python pipeline primitives, it allows security teams to construct real-time, high-throughput pipelines for tasks like log parsing, anomaly detection, digital fingerprinting, and LLM-driven vulnerability analysis.
- **Depends on:**
  - RAPIDS (cuDF, cuML): Provides GPU-accelerated DataFrame manipulations and machine learning tools directly within the pipeline stages.
  - Triton Inference Server: Handles multi-model inference execution, providing dynamic batching and resource pooling for models embedded in the stream.
  - Apache Kafka / Flink (Integration Level): Morpheus supports native Kafka source/sink stages for ingesting and publishing streaming messages. It relies on Kafka for message transport, rather than acting as a full streaming cluster itself.  
  - C++Backend / Python Front End: Uses Pybind11 to offer high-level Python pipeline construction backed by C++ pipeline execution for low-latency throughput.
- **When:** 
  - Ultra-High Throughput Telemetry: Analyzing raw network traffic, PCAP data, or millions of log lines/sec (e.g., via DOCA integration on BlueField DPUs).
  - GPU-Bound Analytics & Real-Time Inference: Pipelines where feature extraction or deep learning (BERT, Graph Neural Networks, LLMs) must run on live data streams with sub-second latency.
  - Digital Fingerprinting (DFP): Generating individual risk models for every user/entity in an enterprise across large windowed datasets.
- **When-not:**
  - Simple Rule-Based ETL: Standard text processing or simple alerting tasks that can easily be handled on CPUs with minimal compute requirements.
  - Massive Distributed Graph Algorithms: Batch graph traversals or graph algorithms that require distributed multi-node CPU clusters rather than streaming GPU pipeline execution.
  - Non-GPU Environments: Deployments lacking NVIDIA GPU acceleration (while Morpheus offers a CPU-only mode, it strips away its core performance advantages).
- **Alternative(s):**
  - Kafka Streams / Apache Flink + Model Server (Triton / TorchServe): The traditional enterprise stack for real-time streaming ETL paired with RPC calls to an external inference endpoint.
  - Ray Streaming + Ray Serve: Python-native distributed task and model serving stream framework.
  - Databricks Structured Streaming / Spark Streaming: Popular for CPU/distributed enterprise batch and streaming pipelines.
- **FinCrime angle:** The GNN Fraud Detection Pipeline in Morpheus targets credit card and bank transfer fraud by capturing network topology instead of evaluating transactions in isolation.   
- **Lab week:** `09-morpheus-fraud`

---

## 4. Done when

- [x] Every framework has a **when** / **when-not** line
- [x] Diagram shows the dependency layers (all edges drawn)
- [x] At least one non-NVIDIA alternative named per serving layer
- [x] The Triton-vs-Dynamo naming question resolved in my own words
- [x] Every card has a **Source checked** date

## 5. Reflection (end of day)

- **What did I build or measure today?**
- **What surprised me or broke, and why?**
- **Production / financial-crime implication?**

## Change log


| Date       | Version | What changed          |
| ---------- | ------- | --------------------- |
| 2026-09-22 | v1      | Initial map (Wk 1 D2) |
| 2026-09-27 | v1.1    | Completed stack diagram: all dependency edges, non-NVIDIA alternatives per layer, diagram notes |


