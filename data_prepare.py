import os
import re
import numpy as np
import pandas as pd
from datasets import DatasetDict
from regex import P
from transformers import (BertTokenizer, 
                          BertModelWithHeads,
                          )
from transformers.adapters.composition import Fuse
import time
import torch

queries_df = pd.read_csv(os.path.join("data", "queries.csv"), sep="\t")
paragraphs_df = pd.read_csv(os.path.join("data", "paragraphs.csv"), sep="\t")

unique_ids = list(set(queries_df["FastAnswerID"]))
print(unique_ids)
print(len(unique_ids))

queries_num = 3
queries_dataframes = []
for fa_id in unique_ids:
    temp_queries_with_ids_df = queries_df[queries_df["FastAnswerID"] == fa_id]
    temp_queries_df = temp_queries_with_ids_df.sample(queries_num)
    queries_dataframes.append(temp_queries_df)

queries_for_test_df = pd.concat(queries_dataframes, axis=0)
queries_for_test_df.to_csv(os.path.join("data", "queries_for_test.csv"), sep="\t", index=False)

queries_with_paragraphs_df = pd.merge(queries_for_test_df, paragraphs_df, on="FastAnswerID", how="outer")
queries_with_paragraphs_df.to_csv(os.path.join("data", "queries_with_paragraphs.csv"), sep="\t", index=False)

print(queries_for_test_df)
print(queries_with_paragraphs_df)