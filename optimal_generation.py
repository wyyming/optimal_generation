# %%
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import networkx as nx
import matplotlib.pyplot as plt
import math

# %%
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

# %%
# expected_text="The quick brown fox jumps over the lazy dog today"
expected_text="Bright sunlight reflects softly across calm ocean waves"
targets = tokenizer(expected_text, return_tensors="pt").input_ids
targets_main=targets[0][1:-1] # targets_main.shape --> [num of token]
tokens=tokenizer.convert_ids_to_tokens(targets_main)
# ['B', 'right', 'Ġsunlight', 'Ġreflects', 'Ġsoftly', 'Ġacross', 'Ġcalm', 'Ġocean', 'Ġwaves', 'Ġat', 'Ġdawn']

block_size=2
blocks=[list(range(i, min(i+block_size, len(tokens)))) for i in range(0, len(tokens), block_size)]
num_blocks=len(blocks)

# %%
# fully masked text
text=" ".join([tokenizer.mask_token]*len(tokens))
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs, labels=targets)
logits_main=outputs.logits[0][1:-1][:] # logits_main.shape --> [num of token, vocab size]
# compute cross entropy for all tokens at once
nll=torch.nn.functional.cross_entropy(logits_main, targets_main,reduction='none')
# transform [num of tokens] to [num of blocks, block size] then sum across columns
remainder = len(tokens) % block_size
if remainder != 0:
    pad_size = block_size - remainder
    nll = torch.cat([nll, torch.zeros(pad_size, device=nll.device)])
blk_logprob = -nll.view(num_blocks, block_size).sum(dim=1)
G=nx.DiGraph()
G.add_node(-1)
for i,lp in enumerate(blk_logprob):
  G.add_edge(-1,i,weight=lp.item())

# %%
for i in range(num_blocks):
  text_list=[]
  for j in range(len(tokens)):
    if block_size*i<=j<block_size*(i+1):
      text_list.append(tokens[j].replace("Ġ", ""))
    else:
      text_list.append(tokenizer.mask_token)
  revealed_text=" ".join(text_list)
  print(revealed_text)

# %%
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
    #   G.add_edge(i,j,weight=float(f"{p.item():.4f}"))
      G.add_edge(i,j,weight=p.item())

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
order = list(nx.dfs_preorder_nodes(mst, source=-1))
print(order)
print(edge_labels)
sum_of_edge=sum(edge_labels.values())
print(math.exp(sum_of_edge))
# 7.038466780220017e-12

# 0:The quick 1:brown fox 2:jumps over 3:the lazy 4:dog today

# %%
# verify probability of this generation order
# TODO: update order so that generation order is correct
total_prob=1
text="[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"
inputs = tokenizer(text, return_tensors="pt")
targets = tokenizer(expected_text, return_tensors="pt").input_ids
outputs = model(**inputs, labels=targets)
initial_ids=blocks[order[1]]
for ids in initial_ids:
    prob=torch.nn.functional.softmax(outputs.logits[0][ids],dim=0)
    total_prob*=prob[targets[0][ids]]

for i in range(1,len(order)-1):
  ids=blocks[order[i]] # [3,4],[1,2]
  masked_words=["[MASK]"]*8
  for j in ids:
    masked_words.insert(j-1,words[j-1])
  text=" ".join(masked_words)
  inputs=tokenizer(text,return_tensors="pt")
  outputs=model(**inputs,labels=targets)
  next_pair=blocks[order[i+1]] # [1,2] [5,6]
  for j in next_pair:
    prob=torch.nn.functional.softmax(outputs.logits[0][j],dim=0)
    total_prob*=prob[targets[0][j]]
# total_prob+=prob[]
print(total_prob)
# 7.0385e-12

# %%
