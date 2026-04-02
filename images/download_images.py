import subprocess
import re
import time
import urllib.parse
import os

queries = {
    "denim_jacket.jpg": "denim-jacket",
    "red_dress.jpg": "red-dress",
    "wool_sweater.jpg": "wool-sweater",
    "leather_belt.jpg": "leather-belt",
    "black_trousers.jpg": "black-trousers",
    "white_shirt.jpg": "white-shirt",
    "green_bomber.jpg": "green-bomber-jacket",
}

for filename, query in queries.items():
    try:
        # Step 1: fetch search page HTML via curl
        url = f"https://unsplash.com/s/photos/{query}"
        result = subprocess.run(
            ["curl", "-s", "-L", url],
            capture_output=True,
            text=True,
            check=True,
        )
        html = result.stdout.replace("&amp;", "&")

        # Step 2: extract first image URL
        matches = re.findall(
            r'https://(?:images|plus)\.unsplash\.com/[^"\'\s\\<>]+',
            html,
        )

        img_url = None
        for m in matches:
            if "photo-" in m or "premium_photo-" in m:
                img_url = m.split(" ")[0]
                break

        if not img_url:
            print(f"No image found for {query}")
            continue

        # Step 3: clean URL and request a reasonable size
        parsed = urllib.parse.urlparse(img_url)
        qs = {"auto": "format", "fit": "crop", "w": "600", "q": "80"}
        new_query = urllib.parse.urlencode(qs)
        img_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"

        # Step 4: download image via curl
        dl_result = subprocess.run(
            ["curl", "-s", "-L", "-o", filename, img_url],
            capture_output=True,
            check=True,
        )
        size = os.path.getsize(filename)
        print(f"Downloaded {filename} ({size} bytes)")

    except Exception as e:
        print(f"Error downloading {filename}: {e}")

    time.sleep(0.5)
