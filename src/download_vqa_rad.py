import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download the VQA-RAD distribution used in this project and save it locally.")
    parser.add_argument("--repo-id", default="flaviagiammarino/vqa-rad")
    parser.add_argument("--output", default="data/vqa_rad")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.repo_id)
    ds.save_to_disk(str(out))
    print(ds)
    print("Saved:", out)


if __name__ == "__main__":
    main()
