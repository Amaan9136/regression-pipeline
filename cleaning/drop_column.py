import pandas as pd

DIRECTORY = "data/"
csv_path = 'Iris.csv'
full_path = DIRECTORY + csv_path
column_to_drop = 'Id'

# Load CSV
df = pd.read_csv(full_path)

# Drop the column if it exists
if column_to_drop in df.columns:
    df = df.drop(columns=[column_to_drop])
    df.to_csv(full_path, index=False)
    print("DROPPED!!!!")
else:
    print(f"Column '{column_to_drop}' not found in {csv_path}.")