import pandas as pd
import os

def load_data(data_folder, file_name) -> pd.DataFrame:
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    return df