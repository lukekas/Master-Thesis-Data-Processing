import os
import requests
from io import BytesIO
from zipfile import ZipFile

numbers = [
    "15/2",
    "16/1",
    "16/2",
    "15/3",
    "15/4",
    "16/3",
    "16/4",
    "17/3",
    "17/4",
    "24/1",
    "24/2",
    "25/1",
    "25/2",
    "26/1",
    "26/2",
    "27/1",
    "27/2",
    "22/4",
    "23/3",
    "23/4",
    "24/3",
    "24/4",
    "25/3",
    "25/4",
    "26/3",
    "26/4",
    "27/3",
    "27/4",
    "28/3",
    "32/2",
    "33/1",
    "33/2",
    "34/1",
    "34/2",
    "35/1",
    "35/2",
    "36/1",
    "36/2",
    "37/1",
    "37/2",
    "38/1",
    "32/4",
    "33/3",
    "33/4",
    "34/3",
    "34/4",
    "35/3",
    "35/4",
    "36/3",
    "36/4",
    "37/3",
    "37/4",
    "38/3",
    "42/2",
    "43/1",
    "43/2",
    "44/1",
    "44/2",
    "45/1",
    "45/2",
    "46/1",
    "46/2",
    "47/1",
    "47/2",
    "48/1",
    "42/4",
    "43/3",
    "43/4",
    "44/3",
    "44/4",
    "45/3",
    "45/4",
    "46/3",
    "46/4",
    "47/3",
    "47/4",
    "48/3",
    "48/4",
    "53/1",
    "53/2",
    "54/1",
    "54/2",
    "55/1",
    "55/2",
    "56/1",
    "56/2",
    "57/1",
    "57/2",
    "58/1",
    "58/2",
    "53/3",
    "53/4",
    "54/3",
    "54/4",
    "55/3",
    "55/4",
    "56/3",
    "56/4",
]


def download_file(url: str, output_path: str) -> None:
    """
    Downloads file, unzips it if it's a zip file, and saves it to the specified output path.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        zip_extension = ".zip"
        if url.endswith(zip_extension):
            with ZipFile(BytesIO(response.content)) as zip_file:
                for file in zip_file.namelist():
                    if file.endswith(".tif"):
                        with zip_file.open(file) as tif_file:
                            with open(
                                output_path.replace(zip_extension, ".tif"), "wb"
                            ) as out_file:
                                out_file.write(tif_file.read())

            print(f"Downloaded and extracted: {output_path}")
            return

        with open(output_path, "wb") as out_file:
            out_file.write(response.content)

        print(f"Downloaded: {output_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    os.makedirs("data/vienna/dsm", exist_ok=True)
    os.makedirs("data/vienna/dgm", exist_ok=True)
    for number in numbers:
        for dm in ["dgm", "dom"]:
            url = f"https://www.wien.gv.at/ma41datenviewer/downloads/geodaten/{dm}_tif/{number.replace('/', '_')}_{dm}_tif.zip"
            if dm == "dom":
                dm = "dsm"
            output_path = f"data/vienna/{dm}/{number.replace('/', '_')}_{dm}_tif.zip"
            download_file(url, output_path)
