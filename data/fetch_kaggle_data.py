import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(dataset_path, download_dir):
    api=KaggleApi()
    api.authenticate()

    print(f"Downloading {dataset_path} to {download_dir}...")
    api.dataset_download_files(dataset_path, path=download_dir, unzip=True)


if __name__ == "__main__":
    dataset_slug = "ahmedshahriarsakib/usa-real-estate-dataset"
    download_path = "data/raw"
    os.makedirs(download_path, exist_ok=True)
    download_dataset(dataset_slug, download_path)


