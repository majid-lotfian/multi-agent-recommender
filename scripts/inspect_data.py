import pandas as pd

laptops_df = pd.read_csv("data/raw/laptops.csv")

print(laptops_df.head(5).to_string(index=False))

print(laptops_df.columns.tolist())

print(len(laptops_df))