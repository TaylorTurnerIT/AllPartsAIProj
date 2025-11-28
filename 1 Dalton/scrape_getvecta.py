import os
import re
import requests
from bs4 import BeautifulSoup

BASE = "https://symbols.radicasoftware.com"
START_PAGE = f"{BASE}/symbols/single-line-symbols"

OUTPUT_DIR = "getvecta_svgs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_page(url):
    print(f"[GET] {url}")
    return requests.get(url, headers={
        "User-Agent": "Mozilla/5.0"
    })

def scrape_stencil_page(url):
    r = get_page(url)
    soup = BeautifulSoup(r.text, "html.parser")

    # Find all <img> tags that reference the CDN
    imgs = soup.find_all("img")
    svg_urls = []

    for img in imgs:
        src = img.get("src")
        if src and "symbols-electrical.getvecta.com" in src:
            if src.endswith(".svg"):
                svg_urls.append(src)

    return svg_urls

def download_svg(url):
    filename = url.split("/")[-1]
    out = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(out):
        return

    print(f"[DL] {filename}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code == 200:
        with open(out, "wb") as f:
            f.write(r.content)
    else:
        print(f"[ERROR] {r.status_code} on {url}")

def get_all_pages():
    r = get_page(START_PAGE)
    soup = BeautifulSoup(r.text, "html.parser")

    # find page links (pagination)
    links = soup.find_all("a", href=True)

    pages = set([START_PAGE])

    for a in links:
        href = a["href"]
        if "single-line-symbols" in href:
            pages.add(BASE + href)

    return sorted(pages)

# ---- RUN SCRAPER ----

all_pages = get_all_pages()
print(f"[INFO] Found {len(all_pages)} pages to scan.")

all_svgs = set()

for page in all_pages:
    svgs = scrape_stencil_page(page)
    print(f"[INFO] Found {len(svgs)} SVGs on page.")
    all_svgs.update(svgs)

# Download all SVGs
for svg in sorted(all_svgs):
    download_svg(svg)

print(f"\nDONE. Downloaded {len(all_svgs)} SVGs.")