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



def predict(premise, hypothesis):
  encoded = tokenizer(premise, hypothesis, return_tensors="pt")
  # if torch.cuda.is_available():
  #  encoded.to("cuda")
  logits = model(**encoded)[0]
  tanh = torch.tanh(logits)
  pred_class = torch.argmax(logits).item()
  print("sigmoid:", torch.sigmoid(logits))
  sigm = max(torch.sigmoid(logits)[0]).item()
  return pred_class, sigm




model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)


adapter_name = "nli_adapter"
mode_name =  "checkpoint-240000"

# adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path = os.path.join(os.getcwd(), "models", mode_name, adapter_name)
model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)

# test_data_df = pd.read_csv(os.path.join("data", "queries_with_answers.csv"), sep="\t")
test_data_df = pd.read_csv(os.path.join("data", "queries_with_paragraphs.csv"), sep="\t")

test_dicts = test_data_df.to_dict(orient="records")


result_dics = []
k = 1
for d in test_dicts:
    q = d["Queries"]
    a = d["Paragraph"]
    # q = d["query"]
    # a = d["answer"]
    t = time.time()
    cls, sgm = predict(q, a)
    d["MouseClass"] = cls
    d["MouseConfidence"] = sgm

    result_dics.append(d)
    print(k, time.time() - t)
    k += 1

result_df = pd.DataFrame(result_dics)
# result_df.to_csv(os.path.join("data", "queries_with_answers_scores.csv"), sep="\t")
result_df.to_csv(os.path.join("data", "queries_with_paragraphs.csv"), sep="\t")