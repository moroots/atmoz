# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 19:56:49 2025

@author: Magnolia
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import subprocess
import concurrent.futures
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

def find_links(url, pattern):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    regex = re.compile(pattern, re.IGNORECASE)
    matching = []

    for link in soup.find_all('a', href=True):
        if regex.search(link['href']) or regex.search(link.text or ""):
            full_url = urljoin(url, link['href'])
            filename = link.text.strip() or Path(link['href']).name
            matching.append((full_url, filename))

    return matching


def curl_download(link, filename, download_folder):
    filepath = download_folder / filename
    # -L follows redirects, -s for silent (no per-file progress)
    result = subprocess.run(
        ["curl", "-L", "-s", "-o", str(filepath), link],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        return True
    else:
        return False


def download_with_curl_parallel(links, download_folder: Path, max_workers=4):
    download_folder.mkdir(parents=True, exist_ok=True)

    # tqdm gives a single progress bar for all downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(curl_download, link, filename, download_folder)
            for link, filename in links
        ]
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Downloading files",
            ncols=80,
        ):
            pass
    return

if __name__ == "__main__":
    url = r"https://www-air.larc.nasa.gov/cgi-bin/ArcView/staqs?SONDE=1"
    pattern = r"Westport.*SONDE"

    links = find_links(url, pattern)
    if not links:
        print("No matching links found.")
    else:
        download_dir = Path("./data/Sondes")
        print(f"Found {len(links)} matching links. Downloading to {download_dir.resolve()}")
        download_with_curl_parallel(links, download_dir, max_workers=6)

