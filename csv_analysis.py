import os
import pandas as pd

df = pd.read_csv(os.path.join("data", "fa_qa_with_val_12538.csv"), sep="\t")
print(df)
df.to_excel(os.path.join("data", "fa_qa_with_val_12538.xlsx"))