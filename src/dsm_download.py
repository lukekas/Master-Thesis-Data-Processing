import re
import os
import requests

meta4_path = "dsm/dsm.meta4"
download_dir = os.path.join("dsm", "downloads")

if __name__ == "__main__":
    os.makedirs(download_dir, exist_ok=True)

    try:
        with open(meta4_path, "r", encoding="utf-8") as f:
            txt = f.read()

        urls = re.findall(r"<url>(.*?)</url>", txt)

        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        if unique_urls:
            print(f"Found {len(unique_urls)} unique URL(s). Starting download...")
            for i, url in enumerate(unique_urls, start=1):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()

                    filename = os.path.basename(url.split("?")[0]) or f"file_{i}"
                    filepath = os.path.join(download_dir, filename)

                    with open(filepath, "wb") as out_file:
                        out_file.write(response.content)

                    print(f"{i}. Downloaded: {filename}")
                except Exception as e:
                    print(f"{i}. Failed to download {url}: {e}")
        else:
            print("No <url>...</url> entries found.")

    except FileNotFoundError:
        print(f"Error: File not found at path '{meta4_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")