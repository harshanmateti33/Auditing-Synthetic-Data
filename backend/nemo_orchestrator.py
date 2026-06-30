# backend/nemo_orchestrator.py
import os
import time
import math
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from metrics import (
    infer_columns,
    agent_coverage,
    agent_privacy,
    agent_fairness,
    agent_utility,
    agent_robustness,
)

# Read NVIDIA API key from environment variable (set in Render dashboard)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

def aggregate_trust(metrics):
    w = {"privacy": 0.25, "fairness": 0.20, "fidelity": 0.20, "utility": 0.25, "robustness": 0.10}
    s = {
        "privacy": max(0.01, metrics["privacy"]["score"]),
        "fairness": max(0.01, metrics["fairness"]["score"]),
        "fidelity": max(0.01, metrics["coverage"]["score"]),
        "utility": max(0.01, metrics["utility"]["score"]),
        "robustness": max(0.01, metrics["robustness"]["score"]),
    }
    
    log_sum = 0.0
    for k, ww in w.items():
        log_sum += ww * math.log(s[k])
        
    return {
        "trust_score": math.exp(log_sum),
        "dimension_scores": s,
        "weights": w
    }

def intent_router_nim(columns, dataset_name):
    print("[NVIDIA NIM API] Calling Intent Router Agent via model 'meta/llama-3.1-nemotron-70b-instruct' on NVIDIA Cloud Catalog API...")
    try:
        schema_summary = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama-3.1-nemotron-70b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a data routing coordinator. Analyze the columns and dataset name, determine the dataset domain (e.g. Healthcare, Finance, Insurance, Generic), and write a one-sentence execution routing decision listing the recommended checks (e.g., Privacy, Fairness, Utility)."},
                    {"role": "user", "content": f"Dataset name: {dataset_name}. Schema: {schema_summary}."}
                ],
                "temperature": 0.2,
                "max_tokens": 100
            },
            timeout=3.0
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[TrustForge] Intent Router NIM call failed: {e}")
        pass
    return "Domain: Generic. Routing execution plan: Privacy, Fairness, and Utility agents activated."

def policy_decision_local(metrics, trust_score):
    checks = [
        {"name": "privacy_score > 0.85", "passed": bool(metrics["privacy"]["score"] > 0.85), "value": metrics["privacy"]["score"], "threshold": 0.85},
        {"name": "fairness_score > 0.90", "passed": bool(metrics["fairness"]["score"] > 0.90), "value": metrics["fairness"]["score"], "threshold": 0.90},
        {"name": "utility_score > 0.90", "passed": bool(metrics["utility"]["score"] > 0.90), "value": metrics["utility"]["score"], "threshold": 0.90},
        {"name": "coverage_score > 0.90", "passed": bool(metrics["coverage"]["score"] > 0.90), "value": metrics["coverage"]["score"], "threshold": 0.90},
        {"name": "robustness_score > 0.95", "passed": bool(metrics["robustness"]["score"] > 0.95), "value": metrics["robustness"]["score"], "threshold": 0.95},
        {"name": "trust_score > 0.88", "passed": bool(trust_score > 0.88), "value": trust_score, "threshold": 0.88},
    ]
    passed = len([c for c in checks if c["passed"]])
    decision = "APPROVE"
    if trust_score < 0.70:
        decision = "REJECT"
    elif passed < len(checks):
        decision = "CONDITIONAL" if trust_score > 0.88 else "REPAIR"
        
    return {
        "decision": decision,
        "checks": checks,
        "passed_count": passed,
        "total_count": len(checks)
    }

def policy_justification_nim_call(metrics, trust_score, decision):
    justification = "Automated compliance metrics validation concluded."
    print("[NVIDIA NIM API] Calling Policy Agent (NeMo Guardrails evaluation reasoning) via model 'meta/llama-3.1-nemotron-70b-instruct' on NVIDIA Cloud Catalog API...")
    try:
        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama-3.1-nemotron-70b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a data compliance checker. Analyze the trust score and decision, and write a single sentence justification explaining why this decision was reached (keep under 30 words, no markdown)."},
                    {"role": "user", "content": f"Trust Score: {trust_score:.3f}. Decision: {decision}. Metrics: Privacy={metrics['privacy']['score']:.2f}, Fairness={metrics['fairness']['score']:.2f}, Utility={metrics['utility']['score']:.2f}."}
                ],
                "temperature": 0.2,
                "max_tokens": 100
            },
            timeout=3.0
        )
        if res.status_code == 200:
            justification = res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[TrustForge] Policy Guardrails NIM call failed: {e}")
        pass
    return justification

def remediation_advisor_nim(suggested_repair):
    print("[NVIDIA NIM API] Calling Repair Agent via model 'meta/llama-3.1-nemotron-70b-instruct' on NVIDIA Cloud Catalog API...")
    try:
        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama-3.1-nemotron-70b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a database repair advisor. Given the suggested repairs, write a concise one-sentence remediation guide for the database administrator."},
                    {"role": "user", "content": f"Remediations to apply: {', '.join(suggested_repair)}."}
                ],
                "temperature": 0.3,
                "max_tokens": 100
            },
            timeout=3.0
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[TrustForge] Remediation NIM call failed: {e}")
    return "Remediation recommended: Apply differential privacy masking, PII scrubbing, and demographic class rebalancing."

def run_nemo_orchestration(df, dataset_name):
    t0 = time.time()
    columns = infer_columns(df)
    audit_log = []
    
    def log_event(msg, agent):
        audit_log.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent": agent,
            "message": msg
        })

    # Start Phase 1: Parallelize Intent Router call with main thread metrics evaluations
    with ThreadPoolExecutor(max_workers=4) as executor:
        log_event(f"Dataset loaded: {len(df)} rows × {len(columns)} columns", "IntentRouter")
        
        # Submit intent router to run in background thread
        router_future = executor.submit(intent_router_nim, columns, dataset_name)
        
        # Compute metrics sequentially on the main thread (they are CPU-bound pandas tasks)
        coverage = agent_coverage(df, columns)
        log_event(f"Coverage Agent completed. Score: {coverage['score']:.3f}", "CoverageAgent")

        privacy = agent_privacy(df, columns)
        log_event(f"Privacy Agent completed. Score: {privacy['score']:.3f}. Detected PII columns: {len(privacy['pii_columns'])}", "PrivacyAgent")

        fairness = agent_fairness(df, columns)
        log_event(f"Fairness Agent completed. Score: {fairness['score']:.3f}. Protected field: '{fairness['protected_attribute']}'", "FairnessAgent")

        utility = agent_utility(df, columns)
        log_event(f"Utility Agent completed. Score (F1): {utility['score']:.3f} using downstream RF classifier.", "UtilityAgent")

        robustness = agent_robustness(df, columns)
        log_event(f"Robustness Agent completed. Accuracy degradation: {robustness['accuracy_degradation']:.3f}", "RobustnessAgent")

        # Resolve the intent router background task
        router_decision = router_future.result()
        log_event(router_decision, "IntentRouter")

        # Step 7: Trust Aggregator Agent
        metrics = {"coverage": coverage, "privacy": privacy, "fairness": fairness, "utility": utility, "robustness": robustness}
        trust = aggregate_trust(metrics)
        log_event(f"Trust Aggregated: τ = {trust['trust_score']:.3f}", "TrustAggregationAgent")

        # Step 8: Policy Agent - calculate decision locally first
        policy = policy_decision_local(metrics, trust["trust_score"])
        
        # Generate suggested repair loop targets
        suggested_repair = []
        if len(privacy["pii_columns"]) > 0:
            suggested_repair.append("mask_pii")
        if privacy["nndr"] < 0.90 or privacy["replica_rate"] > 0.01:
            suggested_repair.append("dp_noise")
        if fairness["spd"] > 0.10:
            suggested_repair.append("balance_minority")
        if any(p["type"] != "identifier_heuristic" for p in privacy["pii_columns"]):
            suggested_repair.append("drop_leaky")

        # Start Phase 2: Parallelize final LLM calls (justification, optional remediation guidance, certification narrative)
        justification_future = executor.submit(policy_justification_nim_call, metrics, trust["trust_score"], policy["decision"])
        
        remediation_future = None
        if policy["decision"] in ["REPAIR", "REJECT"] and len(suggested_repair) > 0:
            remediation_future = executor.submit(remediation_advisor_nim, suggested_repair)
            
        narrative_future = executor.submit(generate_nim_narrative, trust["trust_score"], policy["decision"], len(privacy["pii_columns"]), fairness["protected_attribute"])
        
        # Collect parallel call results
        justification = justification_future.result()
        policy["justification"] = justification
        log_event(f"Policy compliance checklist run. Final decision: {policy['decision']}. Justification: {policy['justification']}", "PolicyAgent")
        
        if remediation_future:
            remediation_text = remediation_future.result()
            log_event(f"Remediation checklist suggested by Repair Agent: {remediation_text}", "RepairAgent")
            
        narrative = narrative_future.result()

    latency_ms = int((time.time() - t0) * 1000)

    # Preview rows
    pdf = df.to_pandas() if hasattr(df, 'to_pandas') else df
    rows_preview = json_rows_preview(pdf)

    return {
        "datasetName": dataset_name,
        "columns": columns,
        "row_count": len(df),
        "metrics": metrics,
        "trust": trust,
        "policy": policy,
        "suggested_repair": suggested_repair,
        "audit_log": audit_log,
        "latency_ms": latency_ms,
        "narrative": narrative,
        "rows_preview": rows_preview
    }

def generate_nim_narrative(trust_score, decision, pii_count, protected_attr):
    print("[NVIDIA NIM API] Calling Certification Agent via model 'meta/llama-3.1-nemotron-70b-instruct' on NVIDIA Cloud Catalog API...")
    try:
        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama-3.1-nemotron-70b-instruct",
                "messages": [
                    {"role": "system", "content": "You are an expert synthetic data auditor. Write a concise, single-sentence technical compliance narrative (max 40 words) summarizing the outcome. Do not use any markdown formatting or lists."},
                    {"role": "user", "content": f"Dataset evaluation complete. Trust Score: {trust_score:.3f}, Policy Decision: {decision}, PII identifiers detected: {pii_count}, Protected attribute analyzed: '{protected_attr}'."}
                ],
                "temperature": 0.5,
                "max_tokens": 150
            },
            timeout=3.0
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[TrustForge] NVIDIA NIM Catalog API call failed: {e}")
        pass
    
    # Fallback to local template-based summary
    status_label = "approved" if decision in ["APPROVE", "CONDITIONAL"] else "flagged for repair"
    return f"TrustForge Auditor Audit: Dataset analysis concluded with a Trust Score of {trust_score:.3f}. The policy check has {status_label} the file based on {pii_count} PII identifiers detected and fairness assessments on protected attribute '{protected_attr}'."

def json_rows_preview(pdf):
    # Prepare preview rows in dictionary formats
    rows = []
    subset = pdf.head(6)
    for _, r in subset.iterrows():
        row_dict = {}
        for col in pdf.columns:
            val = r[col]
            # convert np types to normal python types
            if isinstance(val, (np.integer, np.floating)):
                row_dict[col] = float(val) if isinstance(val, np.floating) else int(val)
            else:
                row_dict[col] = str(val)
        rows.append(row_dict)
    return rows
