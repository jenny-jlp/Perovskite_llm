import os
import requests
import json

def search_cod_perovskites(elements, save_dir):
    """
    Search Crystallography Open Database (COD) for entries containing specified elements
    and download their CIF files.
    
    Args:
        elements (list): List of element symbols, e.g. ['Pb', 'I', 'C', 'H', 'N']
        save_dir (str): Directory where raw CIF files will be saved
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Base URL for COD search API
    base_url = "http://www.crystallography.net/cod/result"
    
    params = {
        "format": "json"
    }
    # Add element parameters el1, el2, ...
    for i, el in enumerate(elements, 1):
        params[f"el{i}"] = el
        
    print(f"Connecting to COD API to search for elements: {elements}...")
    try:
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Error querying COD: HTTP {response.status_code}")
            return
        
        results = response.json()
        print(f"Found {len(results)} matching entries in COD.")
        
        downloaded_count = 0
        for entry in results:
            cod_id = str(entry.get("file"))
            if not cod_id:
                continue
            
            cif_url = f"http://www.crystallography.net/cod/{cod_id}.cif"
            cif_path = os.path.join(save_dir, f"{cod_id}.cif")
            
            # Avoid re-downloading existing files
            if os.path.exists(cif_path):
                continue
                
            print(f"Downloading CIF structure: COD ID {cod_id} -> {cif_path}...")
            try:
                cif_resp = requests.get(cif_url, timeout=15)
                if cif_resp.status_code == 200:
                    with open(cif_path, "w", encoding="utf-8") as f:
                        f.write(cif_resp.text)
                    downloaded_count += 1
                else:
                    print(f"Failed to download COD ID {cod_id}: HTTP {cif_resp.status_code}")
            except Exception as e:
                print(f"Error downloading COD ID {cod_id}: {e}")
                
        print(f"\nDownload completed. Successfully downloaded {downloaded_count} new CIF files to '{save_dir}'.")
        
    except Exception as e:
        print(f"Failed to query COD database: {e}")

if __name__ == "__main__":
    # Standard elements for organic-inorganic lead iodide perovskites (e.g. MAPbI3, FAPbI3)
    target_elements = ["Pb", "I", "C", "H", "N"]
    
    # Save directory inside the project workspace
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cif_save_directory = os.path.join(current_dir, "data", "raw")
    
    search_cod_perovskites(target_elements, cif_save_directory)
