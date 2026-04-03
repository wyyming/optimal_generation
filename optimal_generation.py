# %%
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import networkx as nx
import matplotlib.pyplot as plt

# %%
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

# %%
expected_text="The capital of France is Paris"
words=expected_text.split()
print(words)
text="[MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"

inputs = tokenizer(text, return_tensors="pt")
targets = tokenizer(expected_text, return_tensors="pt").input_ids
outputs = model(**inputs, labels=targets)

print(outputs)
# %%
