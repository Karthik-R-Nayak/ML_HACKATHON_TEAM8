"""Utility script to inspect the contents of any .joblib model or artifact."""
import sys
from pathlib import Path
import joblib
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python inspect_model.py <path_to_joblib_file>")
    sys.exit(1)

path = Path(sys.argv[1]).resolve()
if not path.is_file():
    print(f"Error: File not found: {path}")
    sys.exit(1)

print(f"\n=======================================================")
print(f"Inspecting: {path.name}")
print(f"File Size:  {path.stat().st_size / 1024:.1f} KB")
print(f"=======================================================\n")

obj = joblib.load(path)

if isinstance(obj, dict):
    print("Dictionary Keys:")
    for k, v in obj.items():
        if k == "pipeline":
            print(f"  - {k}: {v.__class__.__name__}")
            for step_name, step_obj in v.named_steps.items():
                print(f"      * Step '{step_name}': {step_obj}")
        elif isinstance(v, (list, tuple)) and len(v) > 5:
            print(f"  - {k}: list of {len(v)} items (e.g. {v[:3]}...)")
        else:
            print(f"  - {k}: {v}")
elif hasattr(obj, "named_steps"):
    print("Scikit-Learn Pipeline Steps:")
    for name, step in obj.named_steps.items():
        print(f"  * {name}: {step}")
else:
    print(f"Object Type: {type(obj)}")
    print(f"Content:\n{obj}")
