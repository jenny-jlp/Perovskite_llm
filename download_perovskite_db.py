import os
import requests
import zipfile

def download_and_extract_zenodo(record_id, save_dir):
    """
    Query the Zenodo API for the given record ID, download the zip file, and extract it.
    
    Args:
        record_id (str): Zenodo record ID (e.g. '5837035')
        save_dir (str): Directory where the files will be saved and extracted
    """
    os.makedirs(save_dir, exist_ok=True)
    
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Querying Zenodo API for record {record_id}...")
    
    try:
        response = requests.get(api_url, timeout=30)
        if response.status_code != 200:
            print(f"Failed to query Zenodo API: HTTP {response.status_code}")
            return
            
        record_data = response.json()
        files = record_data.get("files", [])
        
        if not files:
            print("No files found in this Zenodo record.")
            return
            
        # Find the zip file
        zip_file_info = None
        for f in files:
            # Zenodo API v2 format has links in links['content'] or similar, let's check
            # Usually f['links']['self'] or f['links']['content'] is the download URL
            if f.get("key", "").endswith(".zip") or f.get("filename", "").endswith(".zip"):
                zip_file_info = f
                break
                
        if not zip_file_info:
            # Fallback to the first file if no zip found
            zip_file_info = files[0]
            print(f"No explicit .zip file found, downloading first available file: {zip_file_info.get('key')}")
            
        file_name = zip_file_info.get("key") or zip_file_info.get("filename")
        # Extract only the base name to avoid nested folder write errors
        file_name = os.path.basename(file_name)
        
        download_url = zip_file_info.get("links", {}).get("self") or zip_file_info.get("links", {}).get("content")
        
        if not download_url:
            print("Download URL not found in file metadata.")
            return
            
        zip_path = os.path.join(save_dir, file_name)
        print(f"Downloading file: {file_name} from {download_url}...")
        
        # Download with streaming support to show progress
        with requests.get(download_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
        print(f"Successfully downloaded to {zip_path}")
        
        # Extract if it is a zip file
        if zip_path.endswith(".zip"):
            print(f"Extracting {zip_path} to {save_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            print("Extraction completed successfully!")
            
            # Clean up the zip file to save space
            try:
                os.remove(zip_path)
                print("Temporary ZIP file cleaned up.")
            except Exception as e:
                print(f"Warning: could not delete temporary ZIP file: {e}")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Zenodo record ID for "The Perovskite Database (Archive)" snapshot
    perovskite_db_record = "5837035"
    
    # Save directory inside the project workspace
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_dir, "data")
    
    download_and_extract_zenodo(perovskite_db_record, data_directory)
