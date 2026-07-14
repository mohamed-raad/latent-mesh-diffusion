"""SSL patch for huggingface datasets download."""
import ssl
import urllib3

# Disable SSL verification for urllib3 (used by datasets/requests)
urllib3.disable_warnings()
orig_https = urllib3.HTTPSConnectionPool.ConnectionCls
orig_context = ssl._create_default_https_context
ssl._create_default_https_context = ssl._create_unverified_context

from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
for i, ex in enumerate(ds):
    if i < 2:
        text = ex.get("text", "")
        print(f"Sample {i}: text_len={len(text)}")
    break
print("OK")
