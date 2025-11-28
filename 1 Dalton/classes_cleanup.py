from pathlib import Path
import shutil

CLASSES = Path("dataset/classes")

bad = [
    "128", "256", "512",
    "256_blur", "256_edge", "256_invert", "256_noise", "256_rot-10",
    "256_rot10", "256_rot45", "256_silhouette", "256_thick",
    "512_blur", "512_edge", "512_invert", "512_noise",
    "512_rot-10", "512_rot10", "512_rot45", "512_silhouette", "512_thick"
]

for b in bad:
    d = CLASSES / b
    if d.exists():
        print("Removing:", d)
        shutil.rmtree(d)