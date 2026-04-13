import os

os.environ["HF_HUB_OFFLINE"] = "1"

import json
from pathlib import Path

import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path.home() / "datasets" / "aic_insertions"


# 1. Check file structure
print("=== File Structure ===")
for p in sorted(root.rglob("*")):
    if p.is_file():
        size = p.stat().st_size / 1024
        print(f"  {p.relative_to(root)}  ({size:.1f} KB)")

# 2. Load dataset
print("\n=== Dataset Info ===")
ds = LeRobotDataset(repo_id="my_user/sfp_insertions", root=root)
print(f"  Episodes:  {ds.meta.total_episodes}")
print(f"  Frames:    {ds.meta.total_frames}")
print(f"  FPS:       {ds.meta.fps}")
print(f"  Features:  {list(ds.meta.features.keys())}")

# 3. Check shapes
print("\n=== Sample Frame (index 0) ===")
sample = ds[0]
for key, val in sample.items():
    if hasattr(val, "shape"):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"  {key}: {val}")

# 4. Check observation range
print("\n=== Observation Stats ===")
obs = sample["observation.state"]
print(f"  Shape: {obs.shape}")
print(f"  Min:   {obs.min().item():.4f}")
print(f"  Max:   {obs.max().item():.4f}")
print(f"  Mean:  {obs.mean().item():.4f}")

# 5. Check action range
print("\n=== Action Stats ===")
act = sample["action"]
print(f"  Shape: {act.shape}")
print(f"  Min:   {act.min().item():.4f}")
print(f"  Max:   {act.max().item():.4f}")

# 6. Check a few frames across the episode
print("\n=== Spot Check (first, middle, last frame) ===")
for idx in [0, ds.meta.total_frames // 2, ds.meta.total_frames - 1]:
    s = ds[idx]
    obs_s = s["observation.state"]
    act_s = s["action"]
    # Wrench is last 6 dims of observation
    wrench = obs_s[-6:]
    print(
        f"  Frame {idx}: pos=({obs_s[0]:.3f}, {obs_s[1]:.3f}, {obs_s[2]:.3f})  "
        f"force_z={wrench[2]:.2f}  action_z={act_s[2]:.4f}"
    )

# 7. Check normalization stats
print("\n=== Normalization Stats ===")
stats_path = root / "meta" / "stats.json"
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
    for key in stats:
        s = stats[key]
        if "mean" in s:
            mean_preview = s["mean"][:3] if isinstance(s["mean"], list) else s["mean"]
            print(f"  {key}: mean={mean_preview}...")
else:
    print("  No stats.json found")

# 8. Check custom metadata sidecar
print("\n=== Custom Metadata ===")
sidecar = root / "meta" / "custom_metadata.json"
if sidecar.exists():
    with open(sidecar) as f:
        meta = json.load(f)
    print(json.dumps(meta, indent=2))
else:
    print("  No custom_metadata.json found")

# 9. Read raw parquet for detailed inspection
print("\n=== Raw Parquet Preview ===")
df = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
print(f"  Rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")
print(df.head(3).to_string())
