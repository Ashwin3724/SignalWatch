"""Restore the bundled binary assets from their text-encoded copies."""
from pathlib import Path
import base64
ROOT = Path(__file__).resolve().parent
ASSETS = ['docs/screenshot.png']
for name in ASSETS:
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(Path(str(target) + ".b64").read_text()))
    print("Restored", name)
