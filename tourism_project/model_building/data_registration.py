
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder


def main():
    data_path = Path("tourism_projectfinal/data/tourism.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Make sure tourism.csv is committed in the repo."
        )

    hf_token = os.environ["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"]
    hf_dataset_id = os.environ["Sarathbtech/visit-with-us-tourisms"]

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print("Shape:", df.shape)

    ds = Dataset.from_pandas(df)

    HfFolder.save_token(hf_token)

    api = HfApi(token=hf_token)
    print(f"Pushing dataset to Hugging Face Hub: {hf_dataset_id} ...")
    ds.push_to_hub(hf_dataset_id, token=hf_token)

    print("✅ Dataset registration completed.")


if __name__ == "__main__":
    main()
