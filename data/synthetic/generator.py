"""
data/synthetic/generator.py
---------------------------
Converts Alibaba microservices-v2021 CSVs into compact JSON files.
Falls back to synthetic data when CSVs are not available.

Usage:
    python data/synthetic/generator.py
"""

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).resolve().parents[2]
RAW_DIR   = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

CALL_GRAPH_DIR = RAW_DIR / "MSCallGraph"
RESOURCE_DIR   = RAW_DIR / "MSResource"
RTQPS_DIR      = RAW_DIR / "MSRTQps"

random.seed(42)
np.random.seed(42)


def _find_csv(directory: Path):
    if not directory.exists():
        return None
    for f in sorted(directory.iterdir()):
        if f.suffix == ".csv":
            return f
    return None


def build_service_topology() -> dict:
    csv_path = _find_csv(CALL_GRAPH_DIR)
    if csv_path:
        print(f"[topology] Reading {csv_path.name} ...")
        try:
            import pandas as pd
            df = pd.read_csv(
                csv_path,
                names=["timestamp","traceid","rpcid","um","rpctype","interface","dm","rt"],
                nrows=500_000, on_bad_lines="skip",
            )
            df = df.dropna(subset=["um","dm"])
            df = df[~df["um"].isin(["NAN","(?)",""])]
            df = df[~df["dm"].isin(["NAN","(?)",""])]
            top = df.groupby("um").size().sort_values(ascending=False).head(20).index.tolist()
            edges, seen = [], set()
            for _, row in df[df["um"].isin(top)].iterrows():
                key = (str(row["um"])[:16], str(row["dm"])[:16], str(row["rpctype"]))
                if key in seen: continue
                seen.add(key)
                rt = float(row["rt"]) if str(row["rt"]).lstrip("-").replace(".","").isdigit() else 20.0
                edges.append({"upstream": str(row["um"])[:16], "downstream": str(row["dm"])[:16],
                               "rpc_type": str(row["rpctype"]), "avg_rt_ms": abs(rt)})
                if len(edges) >= 60: break
            print(f"[topology] Extracted {len(edges)} real edges.")
            return {"source": "alibaba-real", "edges": edges}
        except Exception as exc:
            print(f"[topology] CSV parse failed ({exc}), using synthetic.")

    print("[topology] Generating synthetic topology ...")
    edges = [
        {"upstream":"storefront-ui",   "downstream":"api-gateway",     "rpc_type":"http","avg_rt_ms":45},
        {"upstream":"api-gateway",     "downstream":"auth-service",     "rpc_type":"rpc", "avg_rt_ms":18},
        {"upstream":"api-gateway",     "downstream":"payments-api",     "rpc_type":"http","avg_rt_ms":120},
        {"upstream":"api-gateway",     "downstream":"order-service",    "rpc_type":"rpc", "avg_rt_ms":35},
        {"upstream":"payments-api",    "downstream":"payments-db",      "rpc_type":"db",  "avg_rt_ms":8},
        {"upstream":"payments-api",    "downstream":"cache-service",    "rpc_type":"mc",  "avg_rt_ms":3},
        {"upstream":"checkout-service","downstream":"payments-api",     "rpc_type":"rpc", "avg_rt_ms":130},
        {"upstream":"checkout-service","downstream":"inventory-api",    "rpc_type":"rpc", "avg_rt_ms":22},
        {"upstream":"order-service",   "downstream":"inventory-api",    "rpc_type":"rpc", "avg_rt_ms":25},
        {"upstream":"order-service",   "downstream":"notification-svc", "rpc_type":"rpc", "avg_rt_ms":12},
        {"upstream":"auth-service",    "downstream":"user-service",     "rpc_type":"rpc", "avg_rt_ms":15},
        {"upstream":"user-service",    "downstream":"payments-db",      "rpc_type":"db",  "avg_rt_ms":9},
    ]
    return {"source": "synthetic", "edges": edges}


def build_metric_baselines() -> dict:
    csv_path = _find_csv(RESOURCE_DIR)
    if csv_path:
        print(f"[baselines] Reading {csv_path.name} ...")
        try:
            import pandas as pd
            df = pd.read_csv(
                csv_path,
                names=["timestamp","msname","msinstanceid","nodeid","cpu_utilization","memory_utilization"],
                nrows=200_000, on_bad_lines="skip",
            )
            df = df.dropna(subset=["cpu_utilization","memory_utilization"])
            df["cpu_utilization"]    = pd.to_numeric(df["cpu_utilization"],    errors="coerce")
            df["memory_utilization"] = pd.to_numeric(df["memory_utilization"], errors="coerce")
            df = df.dropna()
            top = df.groupby("msname").size().sort_values(ascending=False).head(12).index
            baselines = {}
            for svc in top:
                rows = df[df["msname"] == svc]
                baselines[str(svc)[:16]] = {
                    "cpu_mean":  round(float(rows["cpu_utilization"].mean()), 3),
                    "cpu_std":   round(float(rows["cpu_utilization"].std()),  3),
                    "mem_mean":  round(float(rows["memory_utilization"].mean()), 3),
                    "mem_std":   round(float(rows["memory_utilization"].std()),  3),
                    "cpu_spike": round(float(rows["cpu_utilization"].mean()) + 3*float(rows["cpu_utilization"].std()), 3),
                    "mem_spike": round(float(rows["memory_utilization"].mean()) + 2.5*float(rows["memory_utilization"].std()), 3),
                }
            print(f"[baselines] Extracted {len(baselines)} real service baselines.")
            return {"source": "alibaba-real", "baselines": baselines}
        except Exception as exc:
            print(f"[baselines] CSV parse failed ({exc}), using synthetic.")

    print("[baselines] Generating synthetic baselines ...")
    baselines = {
        "storefront-ui":    {"cpu_mean":0.31,"cpu_std":0.05,"mem_mean":0.45,"mem_std":0.04,"cpu_spike":0.88,"mem_spike":0.91},
        "api-gateway":      {"cpu_mean":0.28,"cpu_std":0.06,"mem_mean":0.40,"mem_std":0.05,"cpu_spike":0.85,"mem_spike":0.88},
        "auth-service":     {"cpu_mean":0.22,"cpu_std":0.04,"mem_mean":0.38,"mem_std":0.03,"cpu_spike":0.79,"mem_spike":0.83},
        "payments-api":     {"cpu_mean":0.35,"cpu_std":0.07,"mem_mean":0.50,"mem_std":0.06,"cpu_spike":0.91,"mem_spike":0.94},
        "payments-db":      {"cpu_mean":0.40,"cpu_std":0.08,"mem_mean":0.60,"mem_std":0.07,"cpu_spike":0.94,"mem_spike":0.97},
        "checkout-service": {"cpu_mean":0.29,"cpu_std":0.05,"mem_mean":0.43,"mem_std":0.04,"cpu_spike":0.84,"mem_spike":0.89},
        "order-service":    {"cpu_mean":0.27,"cpu_std":0.05,"mem_mean":0.41,"mem_std":0.04,"cpu_spike":0.82,"mem_spike":0.87},
        "inventory-api":    {"cpu_mean":0.24,"cpu_std":0.04,"mem_mean":0.39,"mem_std":0.03,"cpu_spike":0.78,"mem_spike":0.82},
        "cache-service":    {"cpu_mean":0.20,"cpu_std":0.03,"mem_mean":0.55,"mem_std":0.05,"cpu_spike":0.75,"mem_spike":0.93},
        "user-service":     {"cpu_mean":0.23,"cpu_std":0.04,"mem_mean":0.37,"mem_std":0.03,"cpu_spike":0.80,"mem_spike":0.81},
        "notification-svc": {"cpu_mean":0.15,"cpu_std":0.03,"mem_mean":0.30,"mem_std":0.02,"cpu_spike":0.65,"mem_spike":0.70},
        "worker-node":      {"cpu_mean":0.55,"cpu_std":0.10,"mem_mean":0.62,"mem_std":0.08,"cpu_spike":0.95,"mem_spike":0.96},
    }
    return {"source": "synthetic", "baselines": baselines}


def build_alert_patterns() -> dict:
    csv_path = _find_csv(RTQPS_DIR)
    if csv_path:
        print(f"[alerts] Reading {csv_path.name} ...")
        try:
            import pandas as pd
            df = pd.read_csv(
                csv_path,
                names=["timestamp","msname","msinstanceid","metrics","value"],
                nrows=200_000, on_bad_lines="skip",
            )
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value","metrics"])
            patterns = {}
            for metric in ["HTTP_RT","consumerRPC_RT","providerRPC_RT","HTTP_MCR","consumerRPC_MCR"]:
                sub = df[df["metrics"] == metric]["value"]
                if len(sub) < 10: continue
                p50 = float(sub.quantile(0.50))
                p95 = float(sub.quantile(0.95))
                p99 = float(sub.quantile(0.99))
                patterns[metric] = {
                    "p50": round(p50,2), "p95": round(p95,2), "p99": round(p99,2),
                    "spike_ratio": round(p99/p50,2) if p50 > 0 else 20.0,
                    "alert_threshold": round(p95,2),
                }
            if patterns:
                print(f"[alerts] Extracted patterns for {list(patterns.keys())}")
                return {"source": "alibaba-real", "patterns": patterns}
        except Exception as exc:
            print(f"[alerts] CSV parse failed ({exc}), using synthetic.")

    print("[alerts] Generating synthetic alert patterns ...")
    patterns = {
        "HTTP_RT":         {"p50":120.0,"p95":450.0,"p99":980.0, "spike_ratio":20.0,"alert_threshold":500.0},
        "consumerRPC_RT":  {"p50":18.0, "p95":85.0, "p99":240.0, "spike_ratio":18.0,"alert_threshold":100.0},
        "providerRPC_RT":  {"p50":15.0, "p95":70.0, "p99":200.0, "spike_ratio":16.0,"alert_threshold":80.0},
        "HTTP_MCR":        {"p50":850.0,"p95":1200.0,"p99":1500.0,"spike_ratio":0.3, "alert_threshold":200.0},
        "consumerRPC_MCR": {"p50":320.0,"p95":480.0, "p99":600.0, "spike_ratio":0.25,"alert_threshold":80.0},
    }
    return {"source": "synthetic", "patterns": patterns}


def main():
    print("=" * 60)
    print("Alibaba Data Processor -> OpenEnv Scenario Generator")
    print("=" * 60)

    topology  = build_service_topology()
    baselines = build_metric_baselines()
    alerts    = build_alert_patterns()

    out_t = PROCESSED / "service_topology.json"
    out_b = PROCESSED / "metric_baselines.json"
    out_a = PROCESSED / "alert_patterns.json"

    out_t.write_text(json.dumps(topology,  indent=2))
    out_b.write_text(json.dumps(baselines, indent=2))
    out_a.write_text(json.dumps(alerts,    indent=2))

    print("\nâœ… Output files written:")
    print(f"   {out_t.name}  ({out_t.stat().st_size // 1024}KB)")
    print(f"   {out_b.name}  ({out_b.stat().st_size // 1024}KB)")
    print(f"   {out_a.name}  ({out_a.stat().st_size // 1024}KB)")
    print(f"\n   Data source: {topology['source']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
