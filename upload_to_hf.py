#!/usr/bin/env python3
"""Upload files to Hugging Face Space via API"""

from huggingface_hub import HfApi, upload_folder
import os

# Your Space ID
SPACE_ID = "tpathak/neural-circuit-designer"

# Path to upload
UPLOAD_PATH = "/Users/tushardhananjaypathak/Desktop/MLEntry/huggingface_deploy"

print("🚀 Uploading to Hugging Face Space...")
print(f"   Space: {SPACE_ID}")
print(f"   Folder: {UPLOAD_PATH}")

# Upload the entire folder
api = HfApi()

try:
    url = upload_folder(
        folder_path=UPLOAD_PATH,
        repo_id=SPACE_ID,
        repo_type="space",
    )
    print(f"\n✅ Upload successful!")
    print(f"🌐 Your app is deploying at: https://huggingface.co/spaces/{SPACE_ID}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 You need to login first. Run:")
    print("   from huggingface_hub import login")
    print("   login()")
