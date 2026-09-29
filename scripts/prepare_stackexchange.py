from pathlib import Path
import json
import sys
import random
import shutil

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

DOMAINS_PER_NODE = 3
DOCS_PER_DOMAIN = 100
SEED = 42

OUTPUT_DIR = Path("nodes") / "data"


def assign_domains_to_nodes():
    """Assign 3 domains to each node."""

    assignments = {}

    for node_id in range(NUM_NODES):
        rng = random.Random(SEED + node_id)
        assignments[node_id] = rng.sample(
            DOMAIN_LIST,
            DOMAINS_PER_NODE
        )

    return assignments


def load_domain(domain):
    """Load one StackExchange domain from HuggingFace."""

    print(f"Loading domain: {domain}")

    dataset = load_dataset(
        DATASET_NAME,
        name=domain,
        trust_remote_code=True,
    )

    return dataset["train"]


def prepare_node_directory(node_id):
    """Create a clean directory for a node."""

    node_dir = OUTPUT_DIR / f"node-{node_id + 1}"

    if node_dir.exists():
        shutil.rmtree(node_dir)

    node_dir.mkdir(parents=True, exist_ok=True)

    return node_dir


def save_node_data(node_id, domains, domain_datasets):
    """
    Combine data from all 3 domains into one data.jsonl file.
    """

    node_dir = prepare_node_directory(node_id)

    output_file = node_dir / "data.jsonl"

    total_docs = 0

    with output_file.open("w", encoding="utf-8") as f:

        for domain in domains:

            dataset = domain_datasets[domain]

            num_docs = min(
                DOCS_PER_DOMAIN,
                len(dataset)
            )

            print(
                f"  Node {node_id + 1} <- "
                f"{domain}: {num_docs} documents"
            )

            for i in range(num_docs):

                row = dataset[i]

                record = {
                    "id": f"{domain}_{i}",
                    "domain": domain,
                    "title_body": row["title_body"],
                    "upvoted_answer": row["upvoted_answer"],
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_docs += 1

    print(
        f"  Created {output_file} "
        f"with {total_docs} documents"
    )


def main():

    assignments = assign_domains_to_nodes()

    print("\nDomain assignments:")
    print("-" * 50)

    for node_id, domains in assignments.items():

        print(
            f"Node {node_id + 1}: "
            f"{', '.join(domains)}"
        )

    print("-" * 50)

    # Cache each domain so the same domain is not downloaded repeatedly.
    domain_cache = {}

    for node_id, domains in assignments.items():

        print(f"\nPreparing Node {node_id + 1}...")

        for domain in domains:

            if domain not in domain_cache:
                domain_cache[domain] = load_domain(domain)

        save_node_data(
            node_id,
            domains,
            domain_cache
        )

    print("\nDataset preparation completed.")


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: "
            "python scripts/prepare_stackexchange.py "
            "<number_of_nodes>"
        )

        sys.exit(1)

    NUM_NODES = int(sys.argv[1])

    if NUM_NODES <= 0:

        print("ERROR: number_of_nodes must be greater than 0.")

        sys.exit(1)

    if NUM_NODES * DOMAINS_PER_NODE > len(DOMAIN_LIST):

        print(
            f"ERROR: Cannot create {NUM_NODES} nodes "
            f"with {DOMAINS_PER_NODE} domains each."
        )

        sys.exit(1)

    main()