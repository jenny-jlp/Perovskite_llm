import os
import pandas as pd

def explore_csv(csv_path):
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    print("\n--- Dataset Summary ---")
    print(f"Total rows (devices): {len(df)}")
    print(f"Total columns (features): {len(df.columns)}")
    
    # 1. Look for composition columns
    comp_cols = [col for col in df.columns if 'composition' in col.lower() or 'formula' in col.lower() or 'short_form' in col.lower()]
    print(f"\nComposition columns: {comp_cols}")
    
    # 2. Check how many records have Cs, FA, MA mixed perovskites
    # In the dataset, the short_form column is often 'Perovskite_composition_short_form'
    # Let's print the top 10 most common composition forms
    if 'Perovskite_composition_short_form' in df.columns:
        print("\nTop 15 most common perovskite compositions:")
        print(df['Perovskite_composition_short_form'].value_counts().head(15))
        
        # Let's count CsFAMA Pb IBr systems
        # A mixed system would typically contain Cs, FA, MA.
        # Let's see if we can find short forms containing 'Cs', 'FA', 'MA' (or Cs, FA, MA cations)
        # Often represented in the database as Cs, FA, MA in 'Perovskite_composition_a_ions' or short_form
        # Let's search for CsFAMA: Cs, FA (Formamidinium, often 'FA'), MA (Methylammonium, often 'MA')
        cs_fa_ma_df = df[
            df['Perovskite_composition_short_form'].str.contains('Cs', na=False) & 
            df['Perovskite_composition_short_form'].str.contains('FA', na=False) & 
            df['Perovskite_composition_short_form'].str.contains('MA', na=False)
        ]
        print(f"\nNumber of triple-cation (CsFAMA) devices: {len(cs_fa_ma_df)}")
        if len(cs_fa_ma_df) > 0:
            print("Sample of triple-cation short forms:")
            print(cs_fa_ma_df['Perovskite_composition_short_form'].value_counts().head(5))
    
    # 3. Check PCE (Power Conversion Efficiency)
    pce_cols = [col for col in df.columns if 'pce' in col.lower() or 'efficiency' in col.lower()]
    print(f"\nPCE columns: {pce_cols}")
    if 'JV_default_PCE' in df.columns:
        print("\nPCE Statistics (JV_default_PCE):")
        print(df['JV_default_PCE'].describe())
        
    # 4. Check layer stack sequence
    print("\nTop 5 Electron Transport Layers (ETL):")
    etl_col = 'ETL_stack_sequence'
    if etl_col in df.columns:
        print(df[etl_col].value_counts().head(5))
        
    print("\nTop 5 Hole Transport Layers (HTL):")
    htl_col = 'HTL_stack_sequence'
    if htl_col in df.columns:
        print(df[htl_col].value_counts().head(5))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        current_dir, 
        "data", 
        "Jesperkemist-perovskitedatabase_data-9b6ed4c", 
        "data", 
        "Perovskite_database_content_all_data.csv"
    )
    explore_csv(csv_path)
