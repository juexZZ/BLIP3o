# from huggingface_hub import snapshot_download

# snapshot_download(repo_id='BLIP3o/BLIP3o-60k', repo_type='dataset')

# download with retry
import time
from huggingface_hub import snapshot_download

def robust_snapshot_download(repo_id, repo_type="model", max_retries=5, wait_seconds=5):
   for attempt in range(1, max_retries + 1):
       try:
           path = snapshot_download(repo_id=repo_id, repo_type=repo_type)
           print(f"Downloaded successfully to: {path}")
           return
       except Exception as e:
           print(f"[Attempt {attempt}] Download failed: {e}")
           if attempt < max_retries:
               print(f"Retrying in {wait_seconds} seconds...")
               time.sleep(wait_seconds)
           else:
               print("Max retries reached. Giving up.")
               raise

if __name__ == "__main__":
   robust_snapshot_download(repo_id="BLIP3o/BLIP3o-Pretrain-Long-Caption", repo_type="dataset", max_retries=10)


# * test data loading
# from datasets import load_dataset
# import glob

# # data_path = "/group-volume/juexiaozhang/hf_cache/hub/datasets--BLIP3o--BLIP3o-Pretrain/snapshots/9c9686108de6074520f5d1c6a74e9b3c8aacd801"
# data_path = "/group-volume/juexiaozhang/hf_cache/hub/datasets--BLIP3o--BLIP3o-Pretrain-Short-Caption/snapshots/e84d184540a6cc4fc3b1ba5528bb45dcc6b3fb0e"
# # data_path = "/group-volume/juexiaozhang/hf_cache/hub/datasets--BLIP3o--BLIP3o-Pretrain-JourneyDB/snapshots/734b390be883b2021ae6d6dc1b24e6b97e7253a2/"
# data_files = glob.glob(f"{data_path}/*.tar")
# train_dataset = load_dataset(
#     "webdataset",
#     data_files=data_files,
#     cache_dir="/group-volume/juexiaozhang/hf_cache",
#     split="train",
#     num_proc=128
# )
# print("len of training data", len(train_dataset))
# train_dataset = train_dataset.rename_column("jpg", "image")
# train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
# train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
# train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
#     ["image", "txt", "type", "image_path"])])
# print(f"finish loading image {len(train_dataset)}")
