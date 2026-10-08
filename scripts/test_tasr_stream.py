import json
import time
import requests

API_URL = "http://localhost:8000/api/v1/query"
NODES_URL = "http://localhost:8000/api/v1/nodes"
API_KEY = "fedrag-dev-key-production-secret-123"
HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}

# Pool of 70 queries sourced from the StackExchange corpus in workspace/data.
# Node -> domain mapping (scripts/prepare_stackexchange.py):
#   node-1=photo, node-2=history, node-3=travel, node-5=biology, node-6=security
# Repeated target-domain queries are used to drive trust divergence.
QUERIES = [
    # 1-15: Biology (Node-5) - clean target domain
    "Why does sympathetic activity constrict pulmonary vessels?",
    "Pedigree Probability of Autosomal Recessive Trait",
    "Why don't different organisms have nucleic acid genomes containing different bases and sugar?",
    "What allows the hypothalamus to detect a lack of thyroid hormones?",
    "Why is chlorophyll green? Isn't there a more energetically favorable color?",
    "How do tea, coffee, and beer dehydrate you?",
    "Does a large effective population size result in faster decay of linkage disequilibrium (LD)?",
    "What does it mean \"specialized populations of neurons\"?",
    "Why humans have the temperature they have?",
    "How long will a typical bacterial strain keep in a -80\u00b0C freezer?",
    "Permeability of Plasma Membrane",
    "Common mistakes when sequencing?",
    "Speciation and Phylogeny of Lactobacillus",
    "Do people with colorblindness have less cones or no cones of a certain type?",
    "Why Do Healing Wounds Feel Warmer To The Touch?",

    # 16-25: Security (Node-6)
    "Why did TLS 1.3 drop AES-CBC?",
    "What is Provable Security?",
    "How to make a simple file integrity checker",
    "How wireless routers are turned into a FlyTrap?",
    "Does manipulating a random password significantly reduce its entropy?",
    "What is the difference between \"unknown\" and \"undefined\" trust?",
    "Are all SQL injections exploitable through time-based attacks?",
    "How do botnets self-propogate exactly?",
    "Is the double submit cookie pattern still effective?",
    "Is my information secure when using Private Tunnel?",

    # 26-35: History (Node-2)
    "Did the Babylonians know the Pythagorean Theorem before Pythagoras formulated it?",
    "How blockadable was the Strait of Gibraltar before Operation \"Torch?\"",
    "Is there evidence that Stalin stopped religious persecution during WW2?",
    "Can the Queen of England fire the prime minister of Australia?",
    "When Lenin met Mussolini, what was Mussolini's impression of Lenin?",
    "Why was the Irish War of Independence in 1918 successful when other revolts failed?",
    "How do vetoes affect re-votes in US Congress?",
    "Did civilian cold war bunkers plan for inclusion of family pets?",
    "What did the average German citizen know about the war?",
    "Did Perry's Black Ships attack Edo harbour during his expedition to Japan?",

    # 36-45: Travel (Node-3) & Photo (Node-1)
    "Chennai or Pune for starting location on India Trip",
    "Is it allowed to take an external hard drive with you on a plane?",
    "6h waiting time in Minneapolis airport, enough for a trip to Mall of America?",
    "Should I give my Japanese host omiyage/gift when meeting them first or when I'm leaving?",
    "Can I enter the US if I have been to Iran?",
    "Where can I buy film for a Kodak Instamatic 500?",
    "Comparable Nikon to the Canon 550d",
    "Reference exposure settings for some common situations",
    "How can I weaken my camera's flash power so it is less disturbing to my subjects?",
    "How do you protect ultrawide angle lenses with a bulbous front element?",

    # 46-55: CROSSING WARM-UP (W=50) -> Rapid Biology re-focus (Node-5)
    "Respiration of an animal cell media",
    "Height prediction based on genetics",
    "Simultaneously using both hands - How does that work?",
    "Could humans choose to establish fibre digesting colonies in our guts?",
    "Characteristics of radius of blood vessel",
    "How can pasteurized milk last 7 days?",
    "How can a species switch from r to K - reproductive strategy in a single generation?",
    "Why is chloride ion classed as a cofactor for amylase rather than as a coenzyme?",
    "Evolution of Heteromorph Ammonites",
    "What is the contribution of viruses to the evolution of mankind?",

    # 56-70: POST-WARMUP ENFORCEMENT -> Continuous targeting of Node-5 (biology) & Node-6 (security)
    "What does these $\\Delta \\Delta G$ numbers signify?",
    "Are some facial features more important than others in human facial recognition?",
    "Why do surface mole tunnels follow directly below the electric fenceline",
    "For Penicillin Binding Proteins, why is the enzyme-peptide complex less stable than the enzyme-\u03b2-lactam complex?",
    "What does \"Activation\" refer to in the context of the symptoms of Schizophrenia and Schizoaffective Disorder?",
    "Introductory book in genetics?",
    "Can you identify this species?",
    "What do I need to do to secure log-in and registration for my website?",
    "How does a security countermeasure failure impact a system?",
    "Executing arbitrary commands through iptables-restore input",
    "Are we safe from phone-to-phone-spreading BlueBorne malware?",
    "Is it possible to exploit computer vision to achieve remote code execution?",
    "Preventing access to encrypted files at all",
    "Should we protect web application source code from being stolen by web hosts through obfuscation?",
    "Vulnerability scan scheduling approach on demand versus change?"
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
