import requests
import re
import os

BASE = "https://symbols-electrical.getvecta.com"
SAVE_DIR = "getvecta_svgs"
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 12.3; rv:104.0) Gecko/20100101 Firefox/104.0"
}

# Stencil ID range – adjust higher if needed (229 exists)
START = 1
END = 400  # scan 1..400 folders

print(f"[*] Scanning stencil folders {START} → {END}")

for stencil_id in range(START, END + 1):
    url = f"{BASE}/stencil_{stencil_id}/"
    print(f"[*] Checking {url}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=5)

        # Skip if blocked, missing, or forbidden
        if r.status_code >= 400:
            continue

        # Extract SVGs from directory listing
        svgs = re.findall(r'href="([^"]+\.svg)"', r.text)

        if not svgs:
            continue

        print(f"    → Found {len(svgs)} SVGs")

        for svg_file in svgs:
            svg_url = f"{url}{svg_file}"
            save_path = os.path.join(SAVE_DIR, svg_file)

            if os.path.exists(save_path):
                continue

            print(f"        Downloading: {svg_file}")
            svg_content = requests.get(svg_url, headers=HEADERS).content

            with open(save_path, "wb") as f:
                f.write(svg_content)

    except Exception as e:
        pass

print("\n[*] DONE — All available SVGs downloaded!")