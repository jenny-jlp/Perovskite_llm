import os
import pandas as pd
import json
import random

def prepare_dataset(csv_path, output_dir):
    """
    Load the perovskite database, filter and clean relevant columns,
    and generate instruction-tuning text templates for LLM training.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading database...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 1. Define columns of interest
    columns = {
        'short_form': 'Perovskite_composition_short_form',
        'long_form': 'Perovskite_composition_long_form',
        'etl': 'ETL_stack_sequence',
        'htl': 'HTL_stack_sequence',
        'backcontact': 'Backcontact_stack_sequence',
        'solvents': 'Perovskite_deposition_solvents',
        'anneal_temp': 'Perovskite_deposition_thermal_annealing_temperature',
        'anneal_time': 'Perovskite_deposition_thermal_annealing_time',
        'pce': 'JV_default_PCE',
        'voc': 'JV_default_Voc',
        'jsc': 'JV_default_Jsc',
        'ff': 'JV_default_FF'
    }
    
    # Filter dataset to rows where PCE is valid
    df_clean = df.dropna(subset=[columns['pce']])
    print(f"Total rows with valid PCE: {len(df_clean)}")
    
    # 2. Convert database rows into training text dialogues
    dataset_entries = []
    
    for idx, row in df_clean.iterrows():
        # Get properties (fill missing values with 'Unknown')
        short_comp = str(row[columns['short_form']]) if pd.notna(row[columns['short_form']]) else "Unknown"
        long_comp = str(row[columns['long_form']]) if pd.notna(row[columns['long_form']]) else "Unknown"
        etl = str(row[columns['etl']]) if pd.notna(row[columns['etl']]) else "Unknown"
        htl = str(row[columns['htl']]) if pd.notna(row[columns['htl']]) else "Unknown"
        backcontact = str(row[columns['backcontact']]) if pd.notna(row[columns['backcontact']]) else "Unknown"
        solvents = str(row[columns['solvents']]) if pd.notna(row[columns['solvents']]) else "Unknown"
        
        temp = f"{row[columns['anneal_temp']]} C" if pd.notna(row[columns['anneal_temp']]) else "Unknown"
        time = f"{row[columns['anneal_time']]} min" if pd.notna(row[columns['anneal_time']]) else "Unknown"
        
        pce = row[columns['pce']]
        voc = f"{row[columns['voc']]} V" if pd.notna(row[columns['voc']]) else "Unknown"
        jsc = f"{row[columns['jsc']]} mA/cm2" if pd.notna(row[columns['jsc']]) else "Unknown"
        ff = f"{row[columns['ff']]} %" if pd.notna(row[columns['ff']]) else "Unknown"
        
        # Construct Prompt (Instruction) and Response (Target)
        prompt = (
            f"Predict the photovoltaic performance of the following perovskite solar cell device.\n"
            f"Device Recipe:\n"
            f"- Perovskite Composition: {long_comp} ({short_comp})\n"
            f"- Electron Transport Layer (ETL): {etl}\n"
            f"- Hole Transport Layer (HTL): {htl}\n"
            f"- Backcontact: {backcontact}\n"
            f"- Deposition Solvents: {solvents}\n"
            f"- Thermal Annealing: {temp} for {time}"
        )
        
        response = (
            f"The predicted photovoltaic performance parameters are:\n"
            f"- Power Conversion Efficiency (PCE): {pce}%\n"
            f"- Open-circuit Voltage (Voc): {voc}\n"
            f"- Short-circuit Current Density (Jsc): {jsc}\n"
            f"- Fill Factor (FF): {ff}"
        )
        
        entry = {
            "instruction": prompt,
            "output": response,
            "is_csfama": "Cs" in short_comp and "FA" in short_comp and "MA" in short_comp
        }
        dataset_entries.append(entry)
        
    # 3. Split into Train/Test sets
    # We want to separate CsFAMA specifically so that we can evaluate performance on it
    csfama_entries = [e for e in dataset_entries if e["is_csfama"]]
    other_entries = [e for e in dataset_entries if not e["is_csfama"]]
    
    print(f"CsFAMA entries count: {len(csfama_entries)}")
    print(f"Other perovskites entries count: {len(other_entries)}")
    
    # Shuffle entries with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(csfama_entries)
    random.shuffle(other_entries)
    
    # Split CsFAMA 80/20 for train/test
    split_csfama = int(len(csfama_entries) * 0.8)
    csfama_train = csfama_entries[:split_csfama]
    csfama_test = csfama_entries[split_csfama:]
    
    # Split other entries 95/5 for train/test
    split_other = int(len(other_entries) * 0.95)
    other_train = other_entries[:split_other]
    other_test = other_entries[split_other:]
    
    # Combine to make final training and testing sets
    train_set = csfama_train + other_train
    test_set = csfama_test + other_test
    
    print(f"Final training set size: {len(train_set)}")
    print(f"Final test set size: {len(test_set)}")
    
    # Save as JSONL files
    train_path = os.path.join(output_dir, "perovskite_llm_train.jsonl")
    test_path = os.path.join(output_dir, "perovskite_llm_test.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for entry in train_set:
            clean_entry = {
                "instruction": entry["instruction"], 
                "output": entry["output"],
                "is_csfama": entry["is_csfama"]
            }
            f.write(json.dumps(clean_entry, ensure_ascii=False) + "\n")
            
    with open(test_path, 'w', encoding='utf-8') as f:
        for entry in test_set:
            clean_entry = {
                "instruction": entry["instruction"], 
                "output": entry["output"], 
                "is_csfama": entry["is_csfama"]
            }
            f.write(json.dumps(clean_entry, ensure_ascii=False) + "\n")
            
    print(f"Saved training dataset to: {train_path}")
    print(f"Saved test dataset to: {test_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        current_dir, 
        "data", 
        "Jesperkemist-perovskitedatabase_data-9b6ed4c", 
        "data", 
        "Perovskite_database_content_all_data.csv"
    )
    output_directory = os.path.join(current_dir, "data", "processed")
    prepare_dataset(csv_path, output_directory)
