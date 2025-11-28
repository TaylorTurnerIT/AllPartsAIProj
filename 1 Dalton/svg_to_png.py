import os
import cairosvg

SVG_DIR = "dataset/final"
PNG_DIR = "dataset/png"

os.makedirs(PNG_DIR, exist_ok=True)

TARGET_SIZE = 512  # change to 256, 1024, etc.

print("[*] Converting SVG → PNG…")

for file in os.listdir(SVG_DIR):
    if file.endswith(".svg"):
        svg_path = os.path.join(SVG_DIR, file)

        # Output name (remove hash if you want)
        png_name = file.replace(".svg", ".png")
        png_path = os.path.join(PNG_DIR, png_name)

        print(f"[*] {file} → {png_name}")

        # Convert
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            output_width=TARGET_SIZE,
            output_height=TARGET_SIZE
        )

print("\n[*] DONE! PNGs saved to dataset/png/")