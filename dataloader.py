import os
import cv2
import numpy as np
import requests
import zipfile
from torch.utils.data import Dataset


class KITTIDataset(Dataset):
    def __init__(self, sequence, base_dir):
        self.sequence = sequence
        self.base_dir = base_dir
        self.sequences_dir = os.path.join(self.base_dir, "dataset", "sequences")
        self.check_and_download_dataset()

    def check_and_download_dataset(self):
        if not os.path.exists(self.sequences_dir):
            print("Dataset not found. Downloading...")
            url = "http://www.cvlibs.net/download.php?file=data_odometry_gray.zip"

            r = requests.get(url)
            with open('data_odometry_gray.zip', 'wb') as f:
                f.write(r.content)
            with zipfile.ZipFile('data_odometry_gray.zip', 'r') as zip_ref:
                zip_ref.extractall(self.base_dir)
            print("Download complete.")

    def __len__(self):
        """Return : dataset length"""
        return len(os.listdir(os.path.join(self.sequences_dir, f"{self.sequence:02d}", "image_0")))

    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        image0_path = os.path.join(self.sequences_dir, f"{self.sequence:02d}", "image_0", f"{idx:06d}.png")
        image1_path = os.path.join(self.sequences_dir, f"{self.sequence:02d}", "image_1", f"{idx:06d}.png")
        try:
            image0 = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
        except Exception:
            image0 = None
            print(f"Could not read {image0_path}")
        try:
            image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        except Exception:
            image1 = None
            print(f"Could not read {image1_path}")

        image0 = cv2.imread(image0_path, cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)

        return image0, image1