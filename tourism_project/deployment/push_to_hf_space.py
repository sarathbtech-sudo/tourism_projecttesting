import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    """
    Push app.py, requirements.txt, and the trained model
    to a Hugging Face Space defined by HF_SPACE_ID.
    """
    hf_token = os.environ["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"]
    space_id = os.environ["Sarathbtech/visit-with-us-tourisms"]  # e.g. "Sarathbtech/visit-with-us-tourism-apps"

    api = HfApi(token=hf_token)

    # Create the Space if it doesn't exist
    print(f"Ensuring Space exists: {space_id}")
    create_repo(
        repo_id=space_id,
        token=hf_token,
        repo_type="space",
        exist_ok=True,
        space_sdk="streamlit",
    )

    # Files to upload
    files_to_upload = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
        ("tourism_project1/model_building/model.pkl", "tourism_project/model_building/model.pkl"),
    ]

    for local_path, remote_path in files_to_upload:
        lp = Path(local_path)
        if not lp.exists():
            raise FileNotFoundError(f"Expected file not found: {lp}")
        print(f"Uploading {lp} to {space_id}:{remote_path}")
        api.upload_file(
            path_or_fileobj=str(lp),
            path_in_repo=remote_path,
            repo_id=space_id,
            repo_type="space",
        )

    print("✅ Files pushed to Hugging Face Space.")


if __name__ == "__main__":
    main()
