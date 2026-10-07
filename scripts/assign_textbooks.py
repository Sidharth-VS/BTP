"""
Assign medical textbooks from dataset/textbooks/en/ to nodes in
workspace/data/<node_id>/.

Each node gets a curated set of textbooks matching the example mapping
(basic sciences, clinical diagnostics, clinical rotations). Run before
starting services so nodes index their assigned books.

Usage:
    python scripts/assign_textbooks.py [workspace_dir] [--force] [--dry-run]
"""

from pathlib import Path
import argparse
import shutil
import sys


SOURCE_DIR = Path("dataset") / "data_clean" / "textbooks" / "en"

TEXTBOOK_EXTENSION = ".txt"

# Book stem (filename without extension) -> node assignment.
# Grouped by subject area:
#   node-15: basic sciences
#   node-16: clinical diagnostics & internal medicine
#   node-17: clinical rotations & board prep
TEXTBOOK_ASSIGNMENT = {
    "node-16": [
        "Anatomy_Gray",
        "Biochemistry_Lippincott",
        "Cell_Biology_Alberts",
        "Histology_Ross",
        "Immunology_Janeway",
        "Physiology_Levy",
    ],
    "node-17": [
        "InternalMed_Harrison",
        "Pathology_Robbins",
        "Pathoma_Husain",
        "Pharmacology_Katzung",
        "Neurology_Adams",
        "Psichiatry_DSM-5",
    ],
    "node-18": [
        "Surgery_Schwartz",
        "Obstentrics_Williams",
        "Gynecology_Novak",
        "Pediatrics_Nelson",
        "First_Aid_Step1",
        "First_Aid_Step2",
    ],
}


def validate_assignment():
    """Check every assigned textbook exists in the source directory."""

    if not SOURCE_DIR.is_dir():
        print(f"ERROR: source directory not found: {SOURCE_DIR}")
        print("Run from the project root.")
        sys.exit(1)

    missing = []

    for node_id, books in TEXTBOOK_ASSIGNMENT.items():

        for book in books:

            src = SOURCE_DIR / f"{book}{TEXTBOOK_EXTENSION}"

            if not src.is_file():
                missing.append(str(src))

    if missing:

        print("ERROR: the following textbooks were not found:")
        print("-" * 50)

        for path in missing:
            print(f"  {path}")

        print("-" * 50)
        sys.exit(1)


def assign_textbooks(workspace_dir, force, dry_run):
    """Copy each textbook into its node's data directory."""

    data_root = Path(workspace_dir) / "data"

    total_copied = 0
    total_skipped = 0

    for node_id, books in TEXTBOOK_ASSIGNMENT.items():

        node_dir = data_root / node_id

        if dry_run:
            print(f"\n[node] {node_id} -> {node_dir} (dry run)")
        else:
            node_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[node] {node_id} -> {node_dir}")

        for book in books:

            src = SOURCE_DIR / f"{book}{TEXTBOOK_EXTENSION}"
            dst = node_dir / src.name

            if dst.exists() and not force:
                print(f"    Skipped {src.name} (already present)")
                total_skipped += 1
                continue

            if dry_run:
                print(f"    Would copy {src.name}")
                total_copied += 1
                continue

            shutil.copy2(src, dst)
            print(f"    Copied {src.name}")
            total_copied += 1

    return total_copied, total_skipped


def main():

    parser = argparse.ArgumentParser(
        description="Assign textbooks from dataset/textbooks/en to workspace nodes."
    )
    parser.add_argument(
        "workspace_dir",
        nargs="?",
        default="./workspace",
        help="Workspace directory (default: ./workspace)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite textbooks already present in node directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing files",
    )

    args = parser.parse_args()

    validate_assignment()

    print("Textbook assignments:")
    print("-" * 50)

    for node_id, books in TEXTBOOK_ASSIGNMENT.items():
        print(f"  {node_id}: {', '.join(books)}")

    print("-" * 50)

    copied, skipped = assign_textbooks(
        args.workspace_dir,
        args.force,
        args.dry_run,
    )

    print()
    print(f"Done. Copied: {copied}, skipped (already present): {skipped}.")

    if not args.dry_run:
        print("Nodes will index these on startup (see nodes/app/main.py).")


if __name__ == "__main__":

    main()
