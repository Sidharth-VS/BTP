import json
import time
import requests

API_URL = "http://localhost:8000/api/v1/query"
NODES_URL = "http://localhost:8000/api/v1/nodes"
API_KEY = "fedrag-dev-key-production-secret-123"
HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}

# Pool of 70 queries: Repeated target-domain queries to test trust divergence
QUERIES = [
    # 1-15: Hardware Security (Node-5) - clean target domain
    "What is the Number Theoretic Transform and why is it used in post-quantum cryptography?",
    "Explain Plantard modular reduction in NTT hardware datapaths.",
    "How does the PipeNTT architecture eliminate memory conflict stalls in FPGA RAM?",
    "What countermeasures defend ML-KEM accelerators against Correlation Power Analysis?",
    "How do AXI4 security wrappers protect against bus probing on Xilinx ZCU102?",
    "Explain the difference between Cooley-Tukey and Gentleman-Sande butterfly units.",
    "What is the modulus q used in ML-KEM compared to ML-DSA?",
    "How does first-order Boolean masking protect lattice cryptographic polynomial operations?",
    "What are twiddle factor ROM addressing patterns in pipelined NTTs?",
    "How does fault injection attack SHAKE sponge permutation rounds in cryptoprocessors?",
    "Why does Montgomery reduction require bit-shifting and constant precomputation?",
    "How does electromagnetic analysis leak secret keys from arithmetic logic units?",
    "What is the mathematical definition of the polynomial ring R_q in ML-KEM?",
    "How do dummy operations decorrelate power consumption traces from secret keys?",
    "What is Barrett reduction and how is it optimized for DSP48E2 slices?",

    # 16-25: Computer Vision (Node-6)
    "How does the Horn-Schunck optical flow method formulate its global smoothness constraint?",
    "What is the difference between Lucas-Kanade local window flow and Horn-Schunck?",
    "How does Laplacian of Gaussian (LoG) isolate multi-scale circular blob structures?",
    "Explain the non-maximum suppression step in the Canny edge detector.",
    "How does the Harris corner detector construct the second-moment autocorrelation matrix?",
    "What textural features are extracted by Gray-Level Co-occurrence Matrices (GLCM)?",
    "Explain double-threshold hysteresis filtering for boundary localization.",
    "How do Local Binary Patterns (LBP) encode micro-textural invariant features?",
    "What is the aperture problem in differential motion estimation?",
    "How does Difference of Gaussians approximate the scale-normalized Laplacian?",

    # 26-35: Medical (Node-2)
    "What are first-line pharmacological treatment protocols for acute hypertension?",
    "What clinical markers differentiate hypertensive emergency from hypertensive urgency?",
    "What are the diagnostic criteria for acute pulmonary edema?",
    "Explain the mechanism of ACE inhibitors in chronic heart failure.",
    "What are secondary causes of renovascular hypertension?",
    "How does nitroprusside reduce afterload in acute hypertensive crises?",
    "What lab investigations evaluate target organ damage in hypertensive crises?",
    "Explain the clinical management of severe pre-eclampsia.",
    "What are the pharmacological contraindications for beta-blockers in asthma?",
    "How is mean arterial pressure calculated in critical care hemodynamic monitoring?",

    # 36-45: Legal (Node-3) & Finance (Node-1)
    "What legal elements establish breach of fiduciary duty under contract law?",
    "What remedies are available upon anticipatory repudiation of a contract?",
    "Explain the doctrine of promissory estoppel in commercial transactions.",
    "What constitutes material breach versus partial breach of contract?",
    "Explain portfolio diversification and the efficient frontier under modern portfolio theory.",
    "How does capital asset pricing model (CAPM) quantify systemic risk via beta?",
    "What is the difference between systematic risk and idiosyncratic unsystematic risk?",
    "Explain the Black-Scholes formula assumptions for European option pricing.",
    "What is value at risk (VaR) in quantitative asset management?",
    "How do bond duration and convexity measure sensitivity to interest rate shifts?",

    # 46-55: CROSSING WARM-UP (W=50) -> Rapid Hardware Security focus (Node-5)
    "How does arithmetic-to-boolean mask conversion prevent side-channel leakage?",
    "Explain the decimation-in-frequency NTT datapath on AMD-Xilinx ZCU104.",
    "What is the impact of clock glitching on ML-DSA signature generation?",
    "Why does Plantard reduction eliminate quotient adjustments in NTT butterfly units?",
    "Explain side-channel leakage detection using Test Vector Leakage Assessment (TVLA).",
    "How do dual-port block RAMs support conflict-free NTT butterfly memory access?",
    "What is the Module-LWE lattice problem underlying Kyber encryption?",
    "How does higher-order masking scale computational complexity in polynomial multiplication?",
    "Explain power consumption modeling using Hamming weight of register states.",
    "How do randomized execution schedules prevent aligned power analysis traces?",

    # 56-70: POST-WARMUP ENFORCEMENT -> Continuous targeting of Node-5 & Node-6
    "What hardware datapath optimizations accelerate NTT polynomial multiplication in Verilog?",
    "Explain the mathematical formulation of the brightness constancy constraint equation.",
    "How do side-channel attacks bypass board-level decoupling capacitors?",
    "Explain scale-space representation using Gaussian blurring in blob detection.",
    "What is the Cooley-Tukey butterfly equation for polynomial ring multiplication?",
    "How does Gauss-Seidel relaxation solve the Horn-Schunck optical flow equation?",
    "What is the primary vulnerability in unprotected SHAKE-256 sponge implementations?",
    "How do spatial gradient operators Sobel and Scharr compute edge orientation?",
    "What is the difference between Boolean and Arithmetic shares in cryptographic masking?",
    "How does Canny edge detection guarantee single-pixel response widths?",
    "Explain the role of twiddle factors in NTT acceleration for post-quantum schemes.",
    "How does GLCM contrast quantify local gray-level variations across an image?",
    "What countermeasures protect FPGA bus interconnects against microprobing?",
    "Explain singular value analysis of the Lucas-Kanade gradient matrix A^T A.",
    "Synthesize the hardware trade-offs between Plantard, Montgomery, and Barrett reduction."
]


def print_table(header, rows):
    col_widths = [len(h) for h in header]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(sep)
    print("| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(header)) + " |")
    print(sep)
    for row in rows:
        print("| " + " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row)) + " |")
    print(sep)


def main():
    print("=" * 80)
    print(" FedRAG: TASR Stream Stress-Test (70 Queries)")
    print(f" Target Endpoint: {API_URL}")
    print(" Warmup Threshold: W = 50 | Cold-Start Horizon: T = 30")
    print("=" * 80)

    # Fetch initial node list
    try:
        res = requests.get(NODES_URL, headers=HEADERS, timeout=10)
        res.raise_for_status()
        initial_nodes = {n["node_id"]: n["trust_score"] for n in res.json().get("nodes", [])}
        print(f"Initial Connected Nodes ({len(initial_nodes)}):")
        for nid, tr in sorted(initial_nodes.items()):
            print(f"  * {nid}: trust_weight = {tr:.3f}")
    except Exception as e:
        print(f"[!] Warning: Could not reach /nodes endpoint: {e}")

    print("\nStarting query stream...\n")
    history = []

    for idx, q_text in enumerate(QUERIES, start=1):
        payload = {"query": q_text, "top_k": 3}
        t0 = time.time()

        try:
            resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=120)
            elapsed = time.time() - t0
            resp.raise_for_status()
            data = resp.json()

            routing_info = data.get("routing_info", {})
            selected = routing_info.get("selected_nodes", [])
            trust_scores = routing_info.get("trust_scores", {})
            strategy = routing_info.get("strategy_used", "UNKNOWN")

            # Phase tag
            if idx <= 50:
                phase_tag = f"Warmup ({idx}/50)"
            else:
                phase_tag = f"Active TASR ({idx}/70)"

            # Compact console log
            node_str = ", ".join(selected)
            score_summary = ", ".join(f"{nid}:{trust_scores.get(nid, 1.0):.3f}" for nid in sorted(trust_scores.keys()))
            print(f"[{idx:02d}/70] [{phase_tag:<16}] {elapsed:.2f}s | Routed: [{node_str}]")
            print(f"       Trust Map: {score_summary}")

            history.append({
                "iter": idx,
                "query": q_text[:40] + "...",
                "selected": node_str,
                "trust": trust_scores
            })

        except requests.exceptions.RequestException as e:
            print(f"[{idx:02d}/70] FAILED: {e}")

        # Brief delay to allow background event loops to settle
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print(" TASR STREAM COMPLETED — FINAL TELEMETRY SUMMARY")
    print("=" * 80)

    # Tabulate snapshot comparisons: Query 1 vs Query 50 vs Query 70
    headers = ["Node ID", "Initial (t=1)", "Pre-Warmup (t=50)", "Final (t=70)", "Net Delta"]
    table_rows = []

    t1_trust = history[0]["trust"] if len(history) >= 1 else {}
    t50_trust = history[49]["trust"] if len(history) >= 50 else {}
    t70_trust = history[-1]["trust"] if history else {}

    all_tracked_nodes = sorted(list(set(list(t1_trust.keys()) + list(t70_trust.keys()))))

    for nid in all_tracked_nodes:
        init_v = t1_trust.get(nid, 0.70)
        w50_v = t50_trust.get(nid, init_v)
        fin_v = t70_trust.get(nid, w50_v)
        delta = fin_v - init_v
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
        table_rows.append([nid, f"{init_v:.4f}", f"{w50_v:.4f}", f"{fin_v:.4f}", delta_str])

    print_table(headers, table_rows)


if __name__ == "__main__":
    main()
