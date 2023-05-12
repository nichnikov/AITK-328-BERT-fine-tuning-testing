import os
import numpy as np
import pandas as pd
from datasets import DatasetDict
from transformers import (BertTokenizer, 
                          BertModelWithHeads,
                          )
from transformers.adapters.composition import Fuse
import torch



def predict(premise, hypothesis):
  encoded = tokenizer(premise, hypothesis, return_tensors="pt")
  # if torch.cuda.is_available():
  #  encoded.to("cuda")
  logits = model(**encoded)[0]
  tanh = torch.tanh(logits)
  pred_class = torch.argmax(logits).item()
  print("sigmoid:", torch.sigmoid(logits))
  return pred_class




model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)


adapter_name = "nli_adapter"
mode_name =  "checkpoint-240000"

# adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path = os.path.join(os.getcwd(), "models", mode_name, adapter_name)
model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)


texts = [("Как учитывать в табеле отпуск по уходу за ребенком когда сотрудник в декрете продолжает работать", 
          "Находясь в отпуске по уходу за ребенком, сотрудник может работать на условиях неполного рабочего времени. Тогда в табеле необходимо отразить оба действия. Чтобы показать отпуск по уходу за ребенком, в коммерческих организациях используют буквенный код «ОЖ» или цифровой код «15». Чтобы отметить в табеле, что сотрудник работает, используют буквенный код «Я» или цифровой код «01». Двойные коды можно проставить через слеш (/) или добавить в табель дополнительную строку (раздел 2 указаний, утв. постановлением Госкомстата от 05.01.2004 № 1, письмо Роструда от 18.03.2008 № 660-6-0). Ниже представлен образец, как заполнять табель учета рабочего времени при работе на условиях неполного рабочего времени в период отпуска по уходу за ребенком.", 1),
          ("можно ли работать в отпуске по уходу за ребенком", 
          "Находясь в отпуске по уходу за ребенком, сотрудник может работать на условиях неполного рабочего времени.", 1)]

k = 1
for q, a, l in texts:
    prd = predict(q, a)
    print(k, "true:", l, "predict:", prd, "val:", l - prd)
    k += 1