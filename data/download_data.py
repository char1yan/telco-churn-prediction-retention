"""Download the raw dataset into data/raw/ (run from the repo root: python data/download_data.py)."""
from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    # IBM sample data set "Telco Customer Churn"
    "Telco-Customer-Churn.csv": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
}

raw = Path(__file__).parent / "raw"
raw.mkdir(exist_ok=True)
for name, url in FILES.items():
    print(f"Downloading {name} ...")
    urlretrieve(url, raw / name)
print("Done.")
