from pathlib import Path
import json
import sys
import random

from datasets import load_dataset


DATASET_NAME = "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl"

DOMAIN_LIST = [
    "electronics",
    "datascience",
    "gaming",
    "academia",
    "chemistry",
    "history",
    "economics",
    "law",
    "cs",
    "biology",
    "mathematica",
    "physics",
    "softwareengineering",
    "security",
    "travel",
    "movies",
    "webapps",
    "gis",
    "android",
    "photo",
]


if len(sys.argv) != 2:
    print("Usage: python scripts/prepare_stackexchange.py <number_of_nodes>")
    sys.exit(1)

NUM_NODES = int(sys.argv[1])

DOMAINS_PER_NODE = 3
DOCS_PER_DOMAIN = 10000

SEED = 42

# IMPORTANT: data goes directly into the existing node folders.
OUTPUT_DIR = Path("nodes") / "data"


def assign_domains_to_nodes():
    assignments = {}

    for node_id in range(NUM_NODES):
        rng = random.Random(SEED + node_id)
        assignments[node_id] = rng.sample(
            DOMAIN_LIST,
            DOMAINS_PER_NODE,
        )

    return assignments


def load_domain(domain):
    print(f"Loading domain: {domain}")

    dataset = load_dataset(
        DATASET_NAME,
        name=domain,
        trust_remote_code=True,
    )

    return dataset["train"]


def save_node_data(node_id, domain, dataset):
    node_dir = OUTPUT_DIR / f"node-{node_id + 1}"
    node_dir.mkdir(parents=True, exist_ok=True)

    output_file = node_dir / f"{domain}.jsonl"

    num_docs = min(DOCS_PER_DOMAIN, len(dataset))

    print(
        f"  node-{node_id + 1} <- {domain}: "
        f"{num_docs} documents"
    )

    with output_file.open("w", encoding="utf-8") as f:
        for i in range(num_docs):
            row = dataset[i]

            record = {
                "id": i,
                "domain": domain,
                "title_body": row["title_body"],
                "upvoted_answer": row["upvoted_answer"],
            }

            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    print(f"Preparing StackExchange data for {NUM_NODES} nodes")
    print(f"Domains per node: {DOMAINS_PER_NODE}")
    print(f"Documents per domain: {DOCS_PER_DOMAIN}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    assignments = assign_domains_to_nodes()

    # Cache each domain so we don't download/load the same
    # StackExchange domain repeatedly if multiple nodes use it.
    domain_cache = {}

    for node_id, domains in assignments.items():
        print(f"Node {node_id + 1}: {domains}")

        for domain in domains:
            if domain not in domain_cache:
                domain_cache[domain] = load_domain(domain)

            save_node_data(
                node_id,
                domain,
                domain_cache[domain],
            )

        print()

    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()