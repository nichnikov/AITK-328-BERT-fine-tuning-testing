import os
import re
from types import GeneratorType
import numpy as np
import pandas as pd
from datasets import (Dataset,
                      DatasetDict)
from regex import P
from transformers import (BertTokenizer, 
                          BertModelWithHeads,
                          TrainingArguments,
                          AdapterTrainer,
                          EvalPrediction)

from transformers.adapters.composition import Fuse
import time
import torch

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


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

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(
      batch["query"],
      batch["answer"],
      max_length=512,
      truncation=True,
      padding="max_length"
  )



model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)


adapter_name = "nli_adapter"
mode_name =  "checkpoint-240000"

# adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path = os.path.join(os.getcwd(), "models", mode_name, adapter_name)
model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)


training_args = TrainingArguments(
    per_device_eval_batch_size=124,
    output_dir="str",
)


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_accuracy,
)

test_data_df = pd.read_csv(os.path.join("data", "chats_testing.csv"), sep="\t")
dataset = Dataset.from_dict({"query": list(test_data_df["Request"]), "answer": list(test_data_df["ShortAnswerText"])})

datasets = DatasetDict({"testing": dataset})
print(datasets)

tokenized_dataset = datasets.map(encode_batch, batched=True)
pred = trainer.predict(tokenized_dataset["testing"])

sigmoid_v = sigmoid(pred.predictions)

test_dicts = test_data_df.to_dict(orient="records")

result_dics = []
for d, sgm in zip(test_dicts, sigmoid_v):
    d["ShortAnswerText"] = re.sub(r"\xa0", " ", d["ShortAnswerText"])
    if sgm[0] > sgm[1]:
        d["MouseClass"] = 0
        d["MouseConfidence"] = sgm[0]
    else:
        d["MouseClass"] = 1
        d["MouseConfidence"] = sgm[1]
    result_dics.append(d)

result_df = pd.DataFrame(result_dics)
result_df.to_csv(os.path.join("data", "chats_checking.csv"), sep="\t")    