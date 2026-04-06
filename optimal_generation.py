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
print(targets)

# %%
blocks = [[1,2],[3,4],[5,6]]
ces=[]
for i in range(len(inputs.input_ids[0])):
  # compute cross entropy for each token position given the expected token ids
  cross_entropy=torch.nn.functional.cross_entropy(outputs.logits[0][i], targets[0][i], reduction='none')
  ces.append(cross_entropy)
# compute cross entropy per block
blk_prob=[]
for blk in blocks:
  blk_nll=0
  for ids in blk:
    blk_nll-=ces[ids]
  blk_prob.append(blk_nll)

print(blk_prob)


# %%
G=nx.DiGraph()
G.add_node(-1)
for i,p in enumerate(blk_prob):
  G.add_edge(-1,i,weight=float(f"{p.item():.4f}"))

# %%
unmasked_blk_prob=[]
for i in range(len(blocks)):
  # unmask one block at a time
  text = ["[MASK]"]*(i*2)
  for ind in blocks[i]:
    text.append(words[ind-1])
  text.extend(["[MASK]"]*((len(blocks)-i-1)*2))
  text=" ".join(text)
  inputs = tokenizer(text, return_tensors="pt")
  outputs = model(**inputs, labels=targets)
  ces=[]
  for j in range(1,len(inputs.input_ids[0])-1):
    # skip unmasked blocks
    if tokenizer.decode(inputs.input_ids[0][j])=="[MASK]":
      cross_entropy=torch.nn.functional.cross_entropy(outputs.logits[0][j], targets[0][j], reduction='none')
      ces.append(cross_entropy)
    else:
      ces.append(None)
  blk_prob=[]
  for j in range(len(blocks)):
    if ces[j*2]!=None:
      blk_prob.append(-ces[j*2]-ces[j*2+1])
    else:
      blk_prob.append(None)
  unmasked_blk_prob.append(blk_prob)
print(unmasked_blk_prob)

# %%
for i,blk in enumerate(unmasked_blk_prob):
  for j,p in enumerate(blk):
    if p!=None:
      G.add_edge(i,j,weight=float(f"{p.item():.4f}"))
pos = nx.circular_layout(G)
nx.draw(G,pos,with_labels=True,connectionstyle="arc3,rad=0.2")
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,connectionstyle="arc3,rad=0.2")
plt.show()
# %%
mst=nx.maximum_spanning_arborescence(G)
nx.draw(mst,pos,with_labels=True)
edge_labels=nx.get_edge_attributes(mst,"weight")
nx.draw_networkx_edge_labels(mst,pos,edge_labels=edge_labels,label_pos=0.4)
plt.show()
# %%
