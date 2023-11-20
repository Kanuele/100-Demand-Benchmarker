import pandas as pd
import os

def import_file(file_path):
    try:
        # Get the file extension
        _, file_extension = os.path.splitext(file_path)
        
        # Check the file extension and import accordingly
        if file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_extension == '.csv':
            df = pd.read_csv(file_path, sep=';')
        else:
            print(f"Unsupported file type: {file_extension}")
            return None
        
        # Do further processing or analysis on the DataFrame if needed
        
        # Return the DataFrame
        return df
    except Exception as e:
        print(f"Error occurred while importing file: {e}")
        return None