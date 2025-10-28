import urllib.request
import os

# Create the target folder if it doesn't exist
folder = "/Users/zhongyitong/Library/Mobile Documents/com~apple~CloudDocs/Internship/MS_Capital/airviz/dataset"
os.makedirs(folder, exist_ok=True)

# List of datasets to download
datasets = [
    ("https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat", "airports.dat"),
    ("https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat", "airlines.dat"),
    ("https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat", "routes.dat"),
    ("https://raw.githubusercontent.com/jpatokal/openflights/master/data/planes.dat", "planes.dat"),
    ("https://raw.githubusercontent.com/jpatokal/openflights/master/data/countries.dat", "countries.dat"),
]

# Download each dataset
for url, filename in datasets:
    file_path = os.path.join(folder, filename)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, file_path)
    print(f"Saved to {file_path}")

print("All datasets downloaded successfully!")