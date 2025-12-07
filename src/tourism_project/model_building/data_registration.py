
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi


def main():
    # Path to raw CSV in the repo (used by GitHub Actions)
    data_path = Path("tourism_project/data/tourism.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Make sure tourism.csv is committed in the repo."
        )

    # Read token and dataset ID from environment variables
    hf_token = os.environ.get("HF_TOKEN")
    hf_dataset_id = os.environ.get("HF_DATASET_ID")

    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Set it as a GitHub secret named 'HF_TOKEN'.")
    if not hf_dataset_id:
        raise ValueError("HF_DATASET_ID is missing. Set it in the workflow env.")

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print("Shape:", df.shape)

    ds = Dataset.from_pandas(df)

    # Initialize API with token
    api = HfApi(token=hf_token)

    # Create dataset repo if it does not exist
    print(f"Ensuring dataset repo '{hf_dataset_id}' exists on Hugging Face Hub ...")
    api.create_repo(
        repo_id=hf_dataset_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    # Push dataset to the Hub
    print(f"Pushing dataset to Hugging Face Hub: {hf_dataset_id} ...")
    ds.push_to_hub(hf_dataset_id, token=hf_token)

    print("✅ Dataset registration completed.")


if __name__ == "__main__":
    main()
