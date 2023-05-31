import os
import re
import time
import numpy as np
import pandas as pd
from datasets import (Dataset,
                      DatasetDict)
from transformers import (BertTokenizer,
                          BertModelWithHeads,
                          TrainingArguments,
                          AdapterTrainer,
                          EvalPrediction)

import torch
from collections import namedtuple


def dict2nmtuple(nt_name: str, dictionary):
    return namedtuple(nt_name, dictionary.keys())(**dictionary)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(premise, hypothesis):
    encoded = tokenizer(premise, hypothesis, return_tensors="pt")
    if torch.cuda.is_available():
        encoded.to("cuda")
    logits = model(**encoded)[0]
    # tanh = torch.tanh(logits)
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
mode_name = "checkpoint-240000"

# adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path = os.path.join(os.getcwd(), "models", mode_name, adapter_name)
model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)

training_args = TrainingArguments(
    per_device_eval_batch_size=150,
    output_dir="str",
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_accuracy,
)

test_data_df = pd.read_csv(os.path.join("data", "EBmodel188655ft_for_testing.csv"), sep="\t")
answers = list(test_data_df["ShortAnswerText"])

query_list = ["по какой форме и в какой срок сдавать заявление о подтверждении основного вида деятельности?",
              "какие отчеты сдавать за 1 квартал 2023"]

results = []
for query in query_list:
    t = time.time()
    queries = [query for i in range(len(answers))]
    dataset = Dataset.from_dict({"query": queries, "answer": answers})

    datasets = DatasetDict({"testing": dataset})

    tokenized_dataset = datasets.map(encode_batch, batched=True)
    pred = trainer.predict(tokenized_dataset["testing"])

    sigmoid_v = sigmoid(pred.predictions)

    test_dicts = [{"Query": q, "Answer": a} for q, a in zip(queries, answers)]

    result_dics = []
    for d, sgm in zip(test_dicts, sigmoid_v):
        d["Answer"] = re.sub(r"\xa0", " ", d["Answer"])
        if sgm[0] > sgm[1]:
            d["MouseClass"] = 0
            d["MouseConfidence"] = sgm[0]
        else:
            d["MouseClass"] = 1
            d["MouseConfidence"] = sgm[1]
        result_dics.append(d)

    nmtpls_1 = [dict2nmtuple("ScoredValue", d) for d in result_dics if d["MouseClass"] == 1]
    nmtpls_0 = [dict2nmtuple("ScoredValue", d) for d in result_dics if d["MouseClass"] == 0]

    if nmtpls_1:
        best_result = sorted(nmtpls_1, key=lambda x: x.MouseConfidence, reverse=True)[0]
        results.append(best_result)

    print(time.time() - t)

print(results)
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(os.path.join("data", "mouse_search_test.csv"), sep="\t")
