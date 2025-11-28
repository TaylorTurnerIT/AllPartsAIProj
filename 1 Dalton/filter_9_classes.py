# file: filter_9_classes.py
import os, shutil, json
from pathlib import Path

CLASS_DIR = Path("dataset/classes")
OUT_DIR = Path("dataset/classes_9")
OUT_DIR.mkdir(exist_ok=True)

KEYMAP = {
    "motor": "motor",
    "electric-motor": "motor",
    "driver-electric-motor": "motor",
    "ac-motor": "motor",

    "switch": "switch",
    "lever-switch": "switch",
    "roller-switch": "switch",
    "flow-switch": "switch",
    "limit-switch": "switch",
    "counter-switch": "switch",

    "transformer": "transformer",
    "current-transformer": "transformer",
    "3-windings": "transformer",

    "fuse": "fuse",

    "circuit-breaker": "breaker",

    "disconnector": "disconnect_switch",
    "isolator": "disconnect_switch",

    "resistor": "resistor",

    "overload": "load_arrow",
    "estop-arrow": "load_arrow",
    "magnetic-overload": "load_arrow",

    "bus-duct": "bus_tie_breaker"
}

def match(name, key):
    return key in name.lower()

for folder in CLASS_DIR.iterdir():
    name = folder.name.lower()

    for key, value in KEYMAP.items():
        if match(name, key):
            out = OUT_DIR / value
            out.mkdir(exist_ok=True)
            for f in folder.glob("*.png"):
                shutil.copy(f, out / f.name)
            print(f"[KEEP] {folder.name} → {value}")
            break

print("[DONE] Filtered 9-class dataset → dataset/classes_9/")