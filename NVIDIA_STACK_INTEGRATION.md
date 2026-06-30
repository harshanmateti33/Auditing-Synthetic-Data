# NVIDIA AI/GPU Stack Integration Guide

This guide details the steps to migrate the simulated JS/Next.js metrics framework in TrustForge to the production-grade **NVIDIA Agentic AI Stack**.

---

## 1. System Architecture Overview

```
                      User Upload (Dataset CSV)
                               │
                               ▼
            Intent Router Agent (NeMo / Nemotron NIM)
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
       ▼                       ▼                       ▼
Coverage Agent           Privacy Agent           Fairness Agent
(RAPIDS cuDF)            (RAPIDS cuDF)            (RAPIDS cuDF)
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                               ▼
                         Utility Agent
                     (RAPIDS cuML XGBoost)
                               │
                               ▼
                        Robustness Agent
                   (TensorRT-LLM / Triton)
                               │
                               ▼
                    Trust Aggregation Agent
                               │
                               ▼
                   Policy Agent (NeMo Guardrails)
                               │
                               ▼
                    Repair Agent (cuDF / NIM)
                               │
                               ▼
                     Certification Agent
                               │
                               ▼
                         Audit Logger
```

---

## 2. Step-by-Step Integration Guide

### Step 1: Agent Orchestration (NeMo Agent Toolkit)
* **Goal**: Replace the local JavaScript sequential loop with an autonomous multi-agent system.
* **Steps**:
  1. Define a Master Coordinator Agent in Python using the **NeMo Agent Toolkit**.
  2. Implement individual Python class workers for each agent (Router, Coverage, Privacy, etc.) inheriting from the `Agent` base class.
  3. Use NeMo's dialog management capabilities to handle parallel execution of the `Coverage`, `Privacy`, and `Fairness` agents, aggregating their returns before calling the `Utility` agent.

### Step 2: GPU-Accelerated Analytics (RAPIDS cuDF & cuML)
* **Goal**: Replace custom JS loop-based calculations (like Naive Bayes, entropy, standard deviations) with GPU-accelerated operations.
* **Steps**:
  1. Import `cudf` in your backend scripts:
     ```python
     import cudf
     ```
  2. Load the synthetic datasets directly into GPU memory via `cudf.read_csv()`.
  3. Replace the Naive Bayes JS classifier with **cuML** GPU-accelerated XGBoost or Random Forest classifiers to calculate downstream F1 and ROC AUC scores.
  4. Perform data perturbations (for the Robustness Agent) and differential privacy noise injections using vectorized cuDF functions.

### Step 3: LLM Reasoning Backbone (NVIDIA NIM & Nemotron)
* **Goal**: Replace external API calls with local, highly optimized NVIDIA Inference Microservices (NIMs).
* **Steps**:
  1. Deploy the `meta/llama-3.1-nemotron-70b-instruct` or `nvidia/nemotron-4-340b-instruct` model inside your infrastructure as a NIM container:
     ```bash
     docker run -d --gpus all \
       -e NGC_API_KEY=$NGC_API_KEY \
       -v /path/to/cache:/opt/nim/.cache \
       -p 8000:8000 \
       nvcr.io/nim/meta/llama-3.1-nemotron-70b-instruct:latest
     ```
  2. Direct the Intent Router Agent and the narrative generator (Certification Agent) to query the local NIM endpoint at `http://localhost:8000/v1/chat/completions`.

### Step 4: Policy Enforcement (NeMo Guardrails)
* **Goal**: Replace basic JS conditional clauses with robust, customizable compliance guardrails.
* **Steps**:
  1. Initialize a NeMo Guardrails configuration directory (`config/`).
  2. Write compliance guidelines inside `config/policies.co` using **Colang**:
     ```colang
     define user express dataset check
       "check this synthetic data"

     define flow dataset policy check
       user express dataset check
       $trust = execute check_trust_score
       if $trust < 0.88
         bot refuse certification
         bot trigger repair loop
     ```
  3. Load the guardrails model on the policy endpoint to act as a wrapper on top of LLM responses and certificate state transitions.

### Step 5: Model Deployment (Triton Inference Server & TensorRT-LLM)
* **Goal**: Accelerate downstream ML evaluation and adversarial noise tests.
* **Steps**:
  1. Compile downstream utility models (XGBoost/MLP) into TensorRT engines for maximum GPU utilization.
  2. Host these models on **Triton Inference Server**.
  3. Serve real-time inference endpoints to the Utility and Robustness agents for quick predictions and noise latency assessments (<15 sec target).

---

## 3. "Not Satisfied" Features Checklist
The following features from the specification are currently simulated in JavaScript and require migration to python/NVIDIA containers to satisfy the specification:

- [ ] **Orchestration Layer**: Migrating agent flow definitions from JS loops to python `nemo-agent-toolkit`.
- [ ] **GPU Execution**: Porting parser/analytical functions to Python `cuDF` (currently uses native Javascript arrays).
- [ ] **Downstream Training**: Upgrading Naive Bayes to RAPIDS-accelerated XGBoost / Random Forest.
- [ ] **Guardrails System**: Transitioning JavaScript threshold conditions to a NeMo Guardrails `.co` runtime engine.
- [ ] **Local Model Hosting**: Deploying a local Llama-3.1-Nemotron NIM container instead of calling an external endpoint.
- [ ] **Service Split**: Transitioning Next.js API Routes to a separate Python FastAPI backend container, running Triton & MongoDB.
