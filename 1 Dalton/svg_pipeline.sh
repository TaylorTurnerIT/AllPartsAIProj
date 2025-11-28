#!/bin/bash

###########################################
# ENHANCED SVG → PNG ML TRAINING PIPELINE
###########################################

INK="/Applications/Inkscape.app/Contents/MacOS/inkscape"  # macOS Inkscape CLI
SVG_DIR="all_svg_symbols"
OUT_DIR="dataset/processed"
LOGFILE="convert_errors.log"

SIZES=(128 256 512 1024)

AUG_DIR="$OUT_DIR/aug"
PNG_DIR="$OUT_DIR/png"

mkdir -p "$AUG_DIR" "$PNG_DIR"

echo "=== SVG Conversion & Augmentation Pipeline ==="
date
echo "Output Directory: $OUT_DIR"
echo "----------------------------------------------"

# Check inkscape exists
if [ ! -f "$INK" ]; then
    echo "[FATAL] Inkscape CLI not found at $INK"
    exit 1
fi

# Helper: run inkscape safely
convert_svg () {
    local svg="$1"
    local size="$2"
    local out="$3"

    "$INK" "$svg" \
        --export-type=png \
        --export-filename="$out" \
        --export-background=white \
        --export-background-opacity=0 \
        -w "$size" -h "$size" 2>>"$LOGFILE"

    if [ ! -f "$out" ]; then
        echo "[ERR] Failed: $svg → $out" | tee -a "$LOGFILE"
    fi
}

# Helper: apply augmentations (uses ImageMagick)
augment_png () {
    local png="$1"
    local base=$(basename "$png" .png)

    # A1: small rotate
    magick "$png" -rotate 10 "$AUG_DIR/${base}_rot10.png"
    magick "$png" -rotate -10 "$AUG_DIR/${base}_rot-10.png"

    # A2: heavy rotate
    magick "$png" -rotate 45 "$AUG_DIR/${base}_rot45.png"

    # A3: thickened strokes
    magick "$png" -morphology Dilate Diamond "$AUG_DIR/${base}_thick.png"

    # A4: silhouette (black fill)
    magick "$png" -alpha extract -fill black -opaque white "$AUG_DIR/${base}_silhouette.png"

    # A5: noise
    magick "$png" -noise 2 "$AUG_DIR/${base}_noise.png"

    # A6: blur
    magick "$png" -blur 0x1 "$AUG_DIR/${base}_blur.png"

    # A7: edge enhancement
    magick "$png" -edge 1 "$AUG_DIR/${base}_edge.png"

    # A8: inverted
    magick "$png" -negate "$AUG_DIR/${base}_invert.png"
}

export -f convert_svg
export -f augment_png
export INK LOGFILE PNG_DIR AUG_DIR

echo "Converting SVGs from $SVG_DIR ..."
sleep 1

############################################
#             MAIN PARALLEL LOOP
############################################

find "$SVG_DIR" -name "*.svg" | while read svg; do

    base=$(basename "$svg" .svg)

    for size in "${SIZES[@]}"; do
        out="$PNG_DIR/${base}_${size}.png"
        echo "[*] Convert: $base.svg → ${size}px"
        convert_svg "$svg" "$size" "$out"

        # Run augmentation for 256px & 512px versions
        if [[ "$size" == "256" || "$size" == "512" ]]; then
            augment_png "$out"
        fi
    done

done

echo "----------------------------------------------"
echo "[DONE] All SVGs converted & augmented!"
echo "Processed SVGs → $OUT_DIR"
echo "Failed conversions logged in $LOGFILE"
date