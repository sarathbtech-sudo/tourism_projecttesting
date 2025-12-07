
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi


def main():
    # Path to the raw dataset inside the repo
    data_path = Path("tourism_projecttestting/data/tourism.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Make sure tourism.csv is committed in the repo."
        )

    # Read token and dataset id from environment (set in GitHub Actions)
    hf_token = os.environ["HF_TOKEN"]
    hf_dataset_id = os.environ["HF_DATASET_ID"]

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print("Shape:", df.shape)

    # Convert to HF Dataset
    ds = Dataset.from_pandas(df)

    # Create dataset repo if it doesn't exist
    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=hf_dataset_id,
        repo_type="dataset",
        exist_ok=True,
    )

    print(f"Pushing dataset to Hugging Face Hub: {hf_dataset_id} ...")
    ds.push_to_hub(hf_dataset_id, token=hf_token)

    print("✅ Dataset registration completed.")


if __name__ == "__main__":
    main()
