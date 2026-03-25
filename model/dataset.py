import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def download_and_store_heart_data(dataset_id=45, filename='heart_disease_raw.csv'):
    """
    Fetches the UCI Heart Disease dataset and stores it as a CSV.
    Reasoning: Serializing to CSV allows for version control and offline access.
    """
    print(f"--- Initiating download for Dataset ID: {dataset_id} ---")
    
    try:
        # 1. Fetch the dataset from the UCI Repository
        # The library returns a 'dot-accessible' dictionary-like object
        heart_disease = fetch_ucirepo(id=dataset_id)
        
        # 2. Extract the 'original' dataframe
        # Reasoning: .data.original includes both features (X) and targets (y)
        # combined in the correct order, preventing alignment errors.
        full_df = heart_disease.data.original
        
        # 3. Store the data locally
        # We use index=False to prevent Pandas from adding an extra 'Unnamed: 0' column
        full_df.to_csv(filename, index=False)
        print(f"Successfully stored dataset to: {os.path.abspath(filename)}")
        
        # 4. Store Metadata (Optional but Recommended for Medical Analysis)
        # Reasoning: Medical datasets use codes (e.g., cp=1,2,3). 
        # Saving the variable descriptions prevents loss of clinical context.
        variables_desc = heart_disease.variables
        variables_desc.to_csv('metadata_variables.csv', index=False)
        print("Successfully stored variable metadata.")
        
        return full_df

    except Exception as e:
        print(f"Error encountered during fetch: {e}")
        return None

# Execute the script
if __name__ == "__main__":
    df = download_and_store_heart_data()
    if df is not None:
        print("\nPreview of the Downloaded Data:")
        print(df.head())