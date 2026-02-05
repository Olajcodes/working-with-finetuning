# import os
# import argparse
# from huggingface_hub import HfApi, create_repo, upload_folder
# from transformers import WhisperProcessor, WhisperForConditionalGeneration

# # ==============================
# # Script to Push a Whisper Checkpoint to Hugging Face Hub
# # ==============================
# # Usage:
# #   python push_checkpoint.py \
# #     --local_dir ./whisper-small-yoruba-129h/checkpoint-8500 \
# #     --repo_id your-username/your-yoruba-whisper-checkpoint-8500 \
# #     --commit_message "Upload checkpoint-8500: WER ~35% on 129h Yoruba data"
# #
# # Prerequisites:
# #   pip install huggingface_hub transformers
# #   huggingface-cli login  # Run once to store your HF token
# # ==============================

# def push_checkpoint(local_dir: str, repo_id: str, commit_message: str = "Upload checkpoint"):
#     """
#     Pushes a local Whisper checkpoint directory to the Hugging Face Hub.
    
#     Args:
#         local_dir: Path to the local checkpoint directory (e.g., ./output/checkpoint-8500)
#         repo_id: Your HF repo ID (e.g., username/whisper-yoruba-checkpoint-8500)
#         commit_message: Optional commit message for the push
#     """
#     if not os.path.exists(local_dir):
#         raise ValueError(f"Local directory {local_dir} does not exist!")

#     # Create the repo if it doesn't exist (private by default; set private=False for public)
#     create_repo(repo_id=repo_id, exist_ok=True, private=True)
#     print(f"Repo {repo_id} ready (created if needed).")

#     # Optional: Load and save processor/model to ensure all files are present
#     # This helps if the checkpoint is missing config.json, preprocessor_config.json, etc.
#     print("Loading model and processor to ensure complete files...")
#     processor = WhisperProcessor.from_pretrained(local_dir)
#     model = WhisperForConditionalGeneration.from_pretrained(local_dir)
    
#     processor.save_pretrained(local_dir)
#     model.save_pretrained(local_dir)
#     print("Processor and model re-saved to local dir.")

#     # Upload the entire folder
#     print(f"Uploading {local_dir} to {repo_id}...")
#     upload_folder(
#         folder_path=local_dir,
#         repo_id=repo_id,
#         commit_message=commit_message,
#         token=True,  # Uses your logged-in token
#     )
#     print(f"Successfully pushed to https://huggingface.co/{repo_id}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Push a Whisper checkpoint to Hugging Face Hub")
#     parser.add_argument("--local_dir", type=str, required=True,
#                         help="Path to local checkpoint directory (e.g., ./whisper-small-yoruba-129h/checkpoint-8500)")
#     parser.add_argument("--repo_id", type=str, required=True,
#                         help="HF repo ID (e.g., your-username/whisper-yoruba-129h-checkpoint-8500)")
#     parser.add_argument("--commit_message", type=str, default="Upload Whisper Yoruba checkpoint",
#                         help="Commit message for the push")

#     args = parser.parse_args()
#     push_checkpoint(args.local_dir, args.repo_id, args.commit_message)


from datasets import load_from_disk, Dataset, DatasetDict
from huggingface_hub import login

# Step 1: Authenticate with Hugging Face
# Option A: Let the script prompt you for your token (recommended for security)
login()

# Option B: Or hardcode your token (less secure, only for testing)
# login(token="hf_your_token_here")

# Step 2: Load your local dataset
# Assuming you saved it to disk using dataset.save_to_disk("path/to/dataset") after training
# Replace the path with where your dataset is stored locally
dataset = load_from_disk("whisper_dataset_yor_183h")  # e.g., "/home/user/whisper_dataset_yor_183h"

# If your dataset is a DatasetDict (e.g., with 'train', 'validation', 'test' splits)
# it will be loaded as DatasetDict automatically by load_from_disk()

# If it's a single Dataset (no splits), you can wrap it if needed:
# dataset = DatasetDict({"train": dataset})

# Step 3: Push to the Hugging Face Hub
# Replace "your_username" with your actual HF username
# The repo will be created automatically if it doesn't exist
dataset.push_to_hub(
    "publica-ai/whisper_dataset_yor_183h",  # repo name on the Hub
    private=True,  # Set to True if you want the dataset to be private
    token=None,     # Uses the token from login() above
    max_shard_size="500MB",  # Recommended for large audio datasets to avoid memory issues
    embed_external_files=True  # Important for audio files — embeds them properly with git-lfs
)

print("Dataset pushed successfully!")