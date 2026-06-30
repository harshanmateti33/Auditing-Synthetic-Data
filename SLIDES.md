# TrustForge: Slide Presentation Deck
**Track A: Agentic Workflows (NVIDIA Agentic AI Stack)**

---

## Slide 1: Title
### TrustForge: Policy-Enforced Autonomous Agentic Compliance & Remediation for Synthetic Enterprise Data

* **Tagline**: Safeguarding the Generative Data Loop with NVIDIA NIM, RAPIDS, and NeMo Guardrails
* **Track**: Track A - Agentic Workflows
* **Team Name**: ForgeGuard AI
* **Team Members**: Harsha Vardhan Mateti & Team
* **Key Focus**: An autonomous multi-agent pipeline designed to evaluate, validate, and repair synthetic datasets before they exit compliance firewalls and enter enterprise model training workflows.

---

## Slide 2: Problem Statement
### The Compliance Bottleneck in Synthetic Data Pipelines

* **The Problem**: Synthetic data is widely adopted to solve data scarcity and protect privacy. However, generated datasets frequently suffer from:
  1. **Low Fidelity/Coverage**: Repetitive distributions that cause model generalization collapse.
  2. **Privacy Leakage**: Exact replicas or near-neighbor memorization that exposes PII (emails, SSNs, credit cards).
  3. **Demographic Bias**: Imbalances in protected attributes (gender, age, race) leading to discriminatory downstream models.
* **Why it Matters**: Non-compliant data causes model degradation, legal liabilities (GDPR, EU AI Act violation), and training pipeline halts.
* **Real-World Impact**: CPU-based statistical analysis halts pipelines for hours on large datasets, and manual remediation script writing slows down database administrators.

---

## Slide 3: Existing Solutions & Research Gap
### Traditional CPU Profiling vs. Autonomous Agentic Verification

* **Existing Solutions**: Packages like SDMetrics, Great Expectations, or pandas-profiling provide static reports.
* **Research Gap**: 
  - They execute entirely on CPUs, creating performance bottlenecks.
  - They lack a policy firewall layer; they notify but do not enforce.
  - They require manual data cleaning; they cannot automatically apply localized mathematical repairs.
  - They present dry metrics instead of clear compliance narratives for non-technical stakeholders.

### Feature Comparison

| Capabilities / Features | Traditional Profiling (e.g. SDMetrics) | TrustForge (Our Solution) |
| :--- | :---: | :---: |
| **Execution Performance** | Slow CPU statistical sweeps | Fast GPU Acceleration via RAPIDS (cuDF/cuML) |
| **Orchestration Layer** | Monolithic sequential scripts | Multi-Agent Autonomous Workflow |
| **Policy Engine** | Manual threshold code | NeMo Guardrails (Colang compliance flows) |
| **Data Remediation** | Manual database scrubbing | Autonomous Repair Agent (differential privacy, masking) |
| **Audit Outputs** | Raw CSV/JSON metrics | Signed Compliance Certificate & NIM Narrative |

---

## Slide 4: Proposed Solution
### TrustForge: Autonomous Policy-Enforced Compliance Firewalls

* **The Innovation**: A multi-agent pipeline backed by the NVIDIA Agentic AI Stack that loads datasets directly in GPU memory to verify and repair data in seconds.
* **Core Value Pillars**:
  1. **Intent-Driven Routing**: LLM inspects data columns and plans custom check flows.
  2. **High-Speed GPU Profiling**: RAPIDS evaluates Coverage, Privacy, Fairness, and Utility on GPUs.
  3. **Guardrail Enforcements**: NeMo Guardrails translates threshold rules into declarative safety flows.
  4. **Self-Healing Loop**: Repair Agent applies targeted remediations (Laplace DP noise, PII scrubbing, class rebalancing) and restarts the evaluation until compliance is achieved.

---

## Slide 5: System Architecture
### End-to-End Orchestration & Data Flow

```
                      +-----------------------------+
                      |  React UI / Next.js Portal  |
                      +--------------+--------------+
                                     |
                         JSON API    | (Proxy Route)
                                     v
                       +-------------+-------------+
                       | FastAPI Python Gateway     |
                       +-------------+-------------+
                                     |
                                     v
                  +------------------+------------------+
                  |  Intent Router Agent (NVIDIA NIM)   |
                  +------------------+------------------+
                                     |
          +--------------------------+--------------------------+
          | (Parallel GPU Analytics - RAPIDS cuDF & cuML)       |
          v                          v                          v
+---------+---------+      +---------+---------+      +---------+---------+
|  Coverage Agent   |      |  Privacy Agent    |      |  Fairness Agent   |
+---------+---------+      +---------+---------+      +---------+---------+
          |                          |                          |
          +--------------------------+--------------------------+
                                     |
                                     v
                  +------------------+------------------+
                  |  Utility Agent (cuML RF Classifier) |
                  +------------------+------------------+
                                     |
                                     v
                  +------------------+------------------+
                  | Robustness Agent (Triton Sim Noise) |
                  +------------------+------------------+
                                     |
                                     v
                  +------------------+------------------+
                  | Trust Aggregation (Geometric Mean)  |
                  +------------------+------------------+
                                     |
                                     v
                  +------------------+------------------+
                  | Policy Agent (NeMo Guardrails .co)  |
                  +------------------+------------------+
                                     |
                 REPAIR / REJECT     |     APPROVE
          +--------------------------+--------------------------+
          v                                                     v
+---------+---------+                                 +---------+---------+
|   Repair Agent    |                                 |CertificationAgent |
| (cuDF Remediate)  |                                 | (NIM Narrative)   |
+---------+---------+                                 +---------+---------+
          |                                                     |
          v                                                     v
   (Rerun Pipeline)                                    +--------+--------+
                                                       |   MongoDB Log   |
                                                       +-----------------+
```

---

## Slide 6: Multi-Agent Workflow
### Role & Communication Pattern of the Agents

1. **Intent Router Agent**: Inspects name/schema; maps data domain (e.g. Healthcare); plans critical check priority using `meta/llama-3.1-70b-instruct`.
2. **Coverage Agent**: Performs Shannon Entropy sweeps on cuDF to detect feature variety.
3. **Privacy Agent**: Calculates duplicate replica rates and Numerical Near Neighbor Distance Ratios (NNDR).
4. **Fairness Agent**: Measures bias on outcomes using Statistical Parity Difference (SPD) and Disparate Impact (DI).
5. **Utility Agent**: Trains a downstream Random Forest classifier in GPU memory via cuML to check ML utility (F1 Score).
6. **Robustness Agent**: Injects Gaussian shifts to check data stability under simulated Triton serving.
7. **Trust Aggregator Agent**: Pools metrics mathematically via weighted geometric mean ($\tau$).
8. **Policy Agent (Guardrail)**: Evaluates $\tau$ and sub-metrics against Colang flow directives to block/pass.
9. **Repair Agent**: Autonomously performs column drops, PII masking, Laplacian noise injection, or minority oversampling using GPU-accelerated cuDF functions.
10. **Certification Agent**: Formulates a clear compliance certificate summary using the NIM Catalog.

---

## Slide 7: NVIDIA Agentic AI Stack
### Deployed Software & Hardware Accelerators

* **RAPIDS cuDF & cuML**: Loads CSVs directly into GPU VRAM. Reduces profiling times from minutes on CPU to sub-second executions. Trains downstream evaluation models via GPU-accelerated Random Forest.
* **NVIDIA NIM (meta/llama-3.1-70b-instruct)**: Provides the reasoning engine for Intent Routing, Guardrail justification, Repair advising, and narrative generation. Hosted locally or queried via NVIDIA Cloud Catalog with low latency.
* **NeMo Guardrails**: Houses the declarative compliance policies (Colang flows) acting as a pipeline firewall.
* **Triton / TensorRT-LLM Simulation**: Simulates distribution shift latency and precision checks on hosted downstream utility classifiers.

---

## Slide 8: Implementation Plan
### Codebase Structure & Phased Rollout

#### Codebase Architecture:
* `app/page.js`: Responsive client dashboard built with Next.js, Shadcn UI, Recharts, and Tailwind CSS.
* `app/api/[[...path]]/route.js`: Next.js endpoint wrapper proxying payloads directly to the Python server.
* `backend/main.py`: FastAPI gateway defining endpoints `/analyze` and `/repair`.
* `backend/nemo_orchestrator.py`: Multi-agent pipeline manager executing LLM calls and pooling metrics.
* `backend/metrics.py`: core calculation engine (using cuDF / pandas and cuML / sklearn fallback).
* `backend/config/`: Colang script configs (`policies.co`) and NeMo model parameters (`config.yml`).

#### Milestones:
* **Phase 1 (Done)**: Implemented front-end pipeline visualizer and charts.
* **Phase 2 (Done)**: Split services, created FastAPI python gateway, ported mathematical calculations to cuDF, and classification training to cuML.
* **Phase 3 (Done)**: Integrated NVIDIA Cloud Catalog completions API and mapped policies inside Colang configuration directory.
* **Phase 4 (Next)**: Compile models into TensorRT engines, host on Triton Inference Server, and establish pipeline webhooks.

---

## Slide 9: Evaluation & Benchmarks
### Metrics & Success Thresholds

* **Trust Score ($\tau$) Formula**:
  $$\tau = (\text{Coverage}^{0.20}) \times (\text{Privacy}^{0.25}) \times (\text{Fairness}^{0.20}) \times (\text{Utility}^{0.25}) \times (\text{Robustness}^{0.10})$$
* **Compliance Benchmarks**:
  - Minimum Trust Score ($\tau$) for Approval: $\ge 0.88$
  - Target Privacy (NNDR): $\ge 0.90$
  - Maximum Fairness Skew (SPD): $\le 0.10$
  - Downstream Utility Classifier (F1): $\ge 0.92$
  - Maximum Robustness Degradation under shift: $\le 0.05$
* **Operational Benchmarks**:
  - Processing Latency: $< 15$ seconds for up to 100k data records (powered by RAPIDS GPU parallelization).
  - Validation Strategy: Rerunning test splits to prove model accuracy did not degrade post-repair.

---

## Slide 10: Demo & Expected Outcome
### End-to-End Enterprise Compliance Workflow

1. **Upload Phase**: User uploads a synthetic dataset (CSV) into the React dashboard.
2. **Analysis Execution**: The UI animates each agent node as the backend performs intent routing, parallel sweeps, utility model fitting, and policy checks.
3. **Interactive Diagnosis**: The Trust Radar visualizes the metric coverage. If any sub-metric falls below the threshold, the Policy Agent flags the run as `REPAIR` or `REJECT`.
4. **Autonomous Self-Healing**: The user selects the recommended actions (e.g. Mask PII, Differential Privacy Noise) and runs the Repair Loop. The agent executes repairs in VRAM and updates metrics.
5. **Certification**: Once trust reaches $\ge 0.88$, a signed, tamper-proof JSON compliance report and narrative certificate are generated and saved to MongoDB.

---

## Slide 11: Roadmap
### Future Enhancements & Scale Strategy

```
           2026-Q3                      2026-Q4                       2027-Q1
+---------------------------+ +---------------------------+ +---------------------------+
| Triton Inference Server   | | Multi-Modal Compliance    | | Enterprise CI/CD Pipeline |
|  - Host utility models on | |  - Support synthetic image| |  - Automated webhook tests |
|    Triton via TensorRT    | |    metadata, audio logs,  | |    integrated with GitHub |
|  - Streamline latency sweeps| |    and tabular datasets   | |    and Airflow pipelines  |
+---------------------------+ +---------------------------+ +---------------------------+
```

---

## Slide 12: References
### Frameworks & NVIDIA Documentation

1. **NVIDIA RAPIDS cuDF & cuML Reference**: [https://docs.rapids.ai/api/cudf/stable/](https://docs.rapids.ai/api/cudf/stable/)
2. **NeMo Guardrails & Colang Flow Language**: [https://github.com/NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
3. **NVIDIA NIM Cloud Catalog Model Specs**: [https://build.nvidia.com/meta/llama-3.1-70b-instruct](https://build.nvidia.com/meta/llama-3.1-70b-instruct)
4. **Differential Privacy in Machine Learning**: Dwork, C. (2006). *Differential Privacy*. 33rd International Colloquium on Automata, Languages and Programming (ICALP).
5. **Shannon Entropy for Feature Divergence Analysis**: Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.
