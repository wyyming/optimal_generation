# %%
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import networkx as nx
import matplotlib.pyplot as plt
import math

from collections import defaultdict

# %%
model_id="answerdotai/ModernBERT-base"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForMaskedLM.from_pretrained(model_id)

# %%
from datasets import load_dataset
ds=load_dataset("roneneldan/TinyStories")

# %%
def find_optimal_gen_order(expected_text,blk_sz=2):
    targets=tokenizer(expected_text, return_tensors="pt").input_ids
    targets_main=targets[0][1:-1] # targets_main.shape --> [num of token]
    num_tokens=targets_main.shape[0]

    blocks=[list(range(i, min(i+blk_sz, num_tokens))) for i in range(0, num_tokens, blk_sz)]
    num_blocks=len(blocks)

    def compute_blk_logprob(input_ids):
        outputs=model(input_ids=input_ids, labels=targets)
        logits_main=outputs.logits[0][1:-1][:] # logits_main.shape --> [num of token, vocab size]
        # compute cross entropy for all tokens at once
        nll=torch.nn.functional.cross_entropy(logits_main, targets_main,reduction='none')
        # transform [num of tokens] to [num of blocks, block size] then sum across columns
        remainder=num_tokens % blk_sz
        if remainder!=0:
            pad_size=blk_sz-remainder
            nll=torch.cat([nll, torch.zeros(pad_size, device=nll.device)])
        blk_logprob=-nll.view(num_blocks, blk_sz).sum(dim=1)
        return blk_logprob

    # fully masked text
    masked_input_ids=targets.clone()
    masked_input_ids[0, 1:-1]=tokenizer.mask_token_id
    blk_logprob=compute_blk_logprob(masked_input_ids)
    G=nx.DiGraph()
    G.add_node(-1)
    for i,lp in enumerate(blk_logprob):
        G.add_edge(-1,i,weight=lp.item())

    # reveal one block at a time
    for i in range(num_blocks):
        revealed_input_ids=masked_input_ids.clone()
        for ind in blocks[i]:
            revealed_input_ids[0][ind+1]=targets[0][ind+1]
        blk_logprob=compute_blk_logprob(revealed_input_ids)
        for j,lp in enumerate(blk_logprob):
            if j!=i:
                G.add_edge(i,j,weight=lp.item())

    # find max spanning arborescence
    pos=nx.circular_layout(G)
    mst=nx.maximum_spanning_arborescence(G)
    # label node with words
    mst.nodes[-1]['label']="start"
    for i,blk in enumerate(blocks):
        words=[tokenizer.decode(targets_main[ind]) for ind in blk]
        mst.nodes[i]['label']=" ".join(words)
    node_labels=nx.get_node_attributes(mst,"label")
    # nx.draw(mst,pos,with_labels=True,labels=node_labels)
    nx.draw(mst,pos,with_labels=True)
    edge_labels=nx.get_edge_attributes(mst,"weight")
    nx.draw_networkx_edge_labels(mst,pos,edge_labels=edge_labels,label_pos=0.4)
    plt.show()
    
    # metrics
    depths=nx.single_source_shortest_path_length(mst, source=-1)
    max_depth=max(depths.values())
    print("depth of tree: ", max_depth)
    levels=defaultdict(list)
    for node, depth in depths.items():
        levels[depth].append(node)
    print("node by levels: ", dict(levels))
    order=list(nx.lexicographical_topological_sort(mst))
    print("generation order: ", order)
    print("sum log likelihood: ", sum(edge_labels.values()))
    
    # find loglikelihood of remembering previous levels
    rmb=nx.DiGraph()
    rmb.add_node(-1)
    level_input_ids=masked_input_ids.clone()
    for nodes in levels.values():
        for n in nodes:
            if n==-1:
                continue
            for ind in blocks[n]:
                level_input_ids[0][ind+1]=targets[0][ind+1]
        blk_logprob=compute_blk_logprob(level_input_ids)
        # keep original generation order
        out_edges=mst.out_edges(nodes)
        for u,v in out_edges:
            rmb.add_edge(u,v,weight=blk_logprob[v].item())
    nx.draw(rmb,pos,with_labels=True)
    rmb_edge_labels=nx.get_edge_attributes(rmb,"weight")
    nx.draw_networkx_edge_labels(rmb,pos,edge_labels=rmb_edge_labels)
    plt.show()
    print("sum log likelihood of remembering: ", sum(rmb_edge_labels.values()))


# %%
text=ds["train"][0]["text"]
# text="The quick brown fox jumps over the lazy dog today"
find_optimal_gen_order(text)

# %%
subset=ds["train"].select(range(10))
for t in subset:
    # print(t["text"])
    find_optimal_gen_order(t["text"])

# %%
# verify probability of this generation order
# TODO: update order so that generation order is correct
total_prob=1
text="[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"
inputs=tokenizer(text, return_tensors="pt")
targets=tokenizer(expected_text, return_tensors="pt").input_ids
outputs=model(**inputs, labels=targets)
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
