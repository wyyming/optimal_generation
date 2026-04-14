import torch
# modernBert
from transformers import AutoTokenizer, AutoModelForMaskedLM

# LLaDA-8B
# from transformers import AutoTokenizer, AutoModelForCausalLM

import networkx as nx
import matplotlib.pyplot as plt

from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("cuda available:", torch.cuda.is_available())

# modernBert
model_id="answerdotai/ModernBERT-base"
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForMaskedLM.from_pretrained(model_id)
model=model.to(device)
# LLaDA-8B
# model = AutoModelForCausalLM.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True, dtype="auto") 
# tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)

from datasets import load_dataset
ds=load_dataset("roneneldan/TinyStories")

def find_optimal_gen_order(expected_text,blk_sz=2):
    targets=tokenizer(expected_text, return_tensors="pt").input_ids.to(device)
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
    plt.figure(figsize=(24,20))
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
    mst=nx.maximum_spanning_arborescence(G)
    pos=nx.bfs_layout(mst,start=-1)
    # label node with words
    mst.nodes[-1]['label']="start"
    for i,blk in enumerate(blocks):
        words=[tokenizer.decode(targets_main[ind]) for ind in blk]
        mst.nodes[i]['label']=" ".join(words)
    node_labels=nx.get_node_attributes(mst,"label")
    nx.draw(mst,pos,with_labels=True,labels=node_labels)
    nx.draw(mst,pos,with_labels=True, font_size=6, node_size=100)
    edge_labels=nx.get_edge_attributes(mst,"weight")
    formatted_labels = {edge: f"{weight:.2f}" for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(mst,pos,edge_labels=formatted_labels,font_size=6)
    plt.text(0.05, 0.95, f"Sum loglik: {sum(edge_labels.values()):.2f}", fontsize=10)
    plt.savefig(f"trees/plot_{blk_sz}.png")

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
    # print("sum log likelihood: ", sum(edge_labels.values()))
    
    # find loglikelihood of remembering previous levels
    plt.figure(figsize=(12,10))
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
    nx.draw(rmb,pos,with_labels=True, font_size=8, node_size=100)
    rmb_edge_labels=nx.get_edge_attributes(rmb,"weight")
    formatted_rmb_edge_labels = {edge: f"{weight:.2f}" for edge, weight in rmb_edge_labels.items()}
    nx.draw_networkx_edge_labels(rmb,pos,edge_labels=formatted_rmb_edge_labels, font_size=6)
    plt.savefig("trees/remember_plot.png")
    print("sum log likelihood of remembering: ", sum(rmb_edge_labels.values()))


text=ds["train"][0]["text"]
# # text="The quick brown fox jumps over the lazy dog today"
for i in range(2,9,2):
    find_optimal_gen_order(text,i)
# find_optimal_gen_order(text)

# subset=ds["train"].select(range(10))
# for t in subset:
#     print(t["text"])
#     find_optimal_gen_order(t["text"])