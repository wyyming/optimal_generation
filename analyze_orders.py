import torch
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from collections import defaultdict
import os
import json
import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment
from datasets import load_dataset

BLOCK_SIZES = [2, 4, 6]
TEXT_INDICES = [0, 1, 2]
K = 5
MASK_TOKEN_ID = 126336

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    'GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device)
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
ds = load_dataset("roneneldan/TinyStories")


# ── Model utilities ───────────────────────────────────────────────────────────

def compute_blk_logprob(input_ids, targets_main, num_blocks, blk_sz):
    num_tokens = targets_main.shape[0]
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0]
    nll = torch.nn.functional.cross_entropy(logits, targets_main, reduction='none')
    remainder = num_tokens % blk_sz
    if remainder != 0:
        nll = torch.cat([nll, torch.zeros(blk_sz - remainder, device=nll.device)])
    return -nll.view(num_blocks, blk_sz).sum(dim=1)


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(targets, blocks, masked_input_ids):
    targets_main = targets[0]
    num_blocks = len(blocks)
    blk_sz = len(blocks[0])

    blk_logprob = compute_blk_logprob(masked_input_ids, targets_main, num_blocks, blk_sz)
    G = nx.DiGraph()
    G.add_node(-1)
    for i, lp in enumerate(blk_logprob):
        G.add_edge(-1, i, weight=lp.item())

    for i in range(num_blocks):
        revealed = masked_input_ids.clone()
        for ind in blocks[i]:
            revealed[0][ind] = targets[0][ind]
        blk_logprob = compute_blk_logprob(revealed, targets_main, num_blocks, blk_sz)
        for j, lp in enumerate(blk_logprob):
            if j != i:
                G.add_edge(i, j, weight=lp.item())

    return G


# ── K-best arborescences ───────────────────────────────────────────────────────

def find_top_k_arborescences(G, k=5):
    seen = []
    results = []
    counter = [0]

    def make_constrained(required, forbidden):
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            if (u, v) in forbidden:
                continue
            w = 1e9 if (u, v) in required else data['weight']
            H.add_edge(u, v, weight=w)
        return H

    def solve(required, forbidden):
        H = make_constrained(required, forbidden)
        try:
            T = nx.maximum_spanning_arborescence(H)
        except nx.NetworkXException:
            return None, -float('inf')
        T_edges = set(T.edges())
        if not required.issubset(T_edges):
            return None, -float('inf')
        weight = sum(G[u][v]['weight'] for u, v in T_edges if G.has_edge(u, v))
        return T, weight

    T, w = solve(frozenset(), frozenset())
    if T is None:
        return []

    counter[0] += 1
    pq = [(-w, counter[0], frozenset(), frozenset(), T)]

    while pq and len(results) < k:
        neg_w, _, required, forbidden, T = heapq.heappop(pq)
        edge_key = frozenset(T.edges())
        if edge_key in seen:
            continue
        seen.append(edge_key)
        results.append((-neg_w, T))

        new_required = set(required)
        for u, v in T.edges():
            if (u, v) not in required:
                sub_T, sub_w = solve(frozenset(new_required), forbidden | {(u, v)})
                if sub_T is not None:
                    counter[0] += 1
                    heapq.heappush(pq, (-sub_w, counter[0], frozenset(new_required), forbidden | {(u, v)}, sub_T))
            new_required.add((u, v))

    return results  # list of (weight, tree)


# ── Likelihood metrics ─────────────────────────────────────────────────────────

def compute_tree_likelihood(tree, G):
    return sum(G[u][v]['weight'] for u, v in tree.edges() if G.has_edge(u, v))


def compute_remember_likelihood(tree, blocks, targets, masked_input_ids):
    targets_main = targets[0]
    num_blocks = len(blocks)
    blk_sz = len(blocks[0])

    depths = nx.single_source_shortest_path_length(tree, source=-1)
    levels = defaultdict(list)
    for node, depth in depths.items():
        levels[depth].append(node)

    level_input_ids = masked_input_ids.clone()
    total = 0.0

    for depth in sorted(levels.keys()):
        nodes = levels[depth]
        for n in nodes:
            if n == -1:
                continue
            for ind in blocks[n]:
                level_input_ids[0][ind] = targets[0][ind]

        blk_logprob = compute_blk_logprob(level_input_ids, targets_main, num_blocks, blk_sz)
        for n in nodes:
            for _, v in tree.out_edges(n):
                total += blk_logprob[v].item()

    return total


# ── LTR tree ──────────────────────────────────────────────────────────────────

def build_ltr_tree(num_blocks):
    T = nx.DiGraph()
    T.add_node(-1)
    T.add_edge(-1, 0)
    for i in range(num_blocks - 1):
        T.add_edge(i, i + 1)
    return T


# ── Unordered tree-edit distance ──────────────────────────────────────────────

def _subtree_size(tree, node, cache):
    if node in cache:
        return cache[node]
    size = 1 + sum(_subtree_size(tree, c, cache) for c in tree.successors(node))
    cache[node] = size
    return size


def unordered_ted(T1, T2):
    """Unordered tree edit distance between T1 and T2, both rooted at -1."""
    size1, size2 = {}, {}
    for n in T1.nodes():
        _subtree_size(T1, n, size1)
    for n in T2.nodes():
        _subtree_size(T2, n, size2)

    memo = {}

    def ted(u, v):
        if (u, v) in memo:
            return memo[(u, v)]

        relabel = 0 if u == v else 1
        c1 = list(T1.successors(u))
        c2 = list(T2.successors(v))

        if not c1 and not c2:
            result = relabel
        elif not c1:
            result = relabel + sum(size2[c] for c in c2)
        elif not c2:
            result = relabel + sum(size1[c] for c in c1)
        else:
            n1, n2 = len(c1), len(c2)
            n = n1 + n2
            cost = np.zeros((n, n))
            for i, ci in enumerate(c1):
                for j, dj in enumerate(c2):
                    cost[i][j] = ted(ci, dj)
                for j in range(n2, n):
                    cost[i][j] = size1[ci]
            for j, dj in enumerate(c2):
                for i in range(n1, n):
                    cost[i][j] = size2[dj]
            row, col = linear_sum_assignment(cost)
            result = relabel + float(cost[row, col].sum())

        memo[(u, v)] = result
        return result

    return int(round(ted(-1, -1)))


# ── Tree serialization ────────────────────────────────────────────────────────

def tree_to_dict(tree, blocks, tokenizer, targets_main, node=-1):
    info = {'id': node, 'label': 'start' if node == -1 else
            ''.join(tokenizer.decode(targets_main[ind]) for ind in blocks[node])}
    children = sorted(tree.successors(node))
    if children:
        info['children'] = [tree_to_dict(tree, blocks, tokenizer, targets_main, c) for c in children]
    return info


def get_generation_order(tree):
    return [n for n in nx.lexicographical_topological_sort(tree) if n != -1]


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_one(text_idx, text, blk_sz):
    print(f"  [text={text_idx}, blk_sz={blk_sz}] tokenizing...")
    targets = tokenizer(text, return_tensors="pt").input_ids.to(device)
    targets_main = targets[0]
    num_tokens = targets_main.shape[0]

    blocks = [list(range(i, min(i + blk_sz, num_tokens))) for i in range(0, num_tokens, blk_sz)]
    num_blocks = len(blocks)

    masked_input_ids = targets.clone()
    masked_input_ids[0][:] = MASK_TOKEN_ID

    block_labels = [''.join(tokenizer.decode(targets_main[ind]) for ind in blk) for blk in blocks]

    print(f"  [text={text_idx}, blk_sz={blk_sz}] building graph ({num_blocks} blocks)...")
    G = build_graph(targets, blocks, masked_input_ids)

    print(f"  [text={text_idx}, blk_sz={blk_sz}] finding top-{K} arborescences...")
    top_k = find_top_k_arborescences(G, k=K)

    ltr_tree = build_ltr_tree(num_blocks)

    arborescences = []
    for rank, (weight, tree) in enumerate(top_k):
        print(f"  [text={text_idx}, blk_sz={blk_sz}] computing metrics for rank {rank+1}...")
        rmb = compute_remember_likelihood(tree, blocks, targets, masked_input_ids)
        depths = nx.single_source_shortest_path_length(tree, source=-1)
        depth = max(depths.values())
        ted = unordered_ted(tree, ltr_tree)
        order = get_generation_order(tree)
        arborescences.append({
            'rank': rank + 1,
            'tree_likelihood': round(weight, 4),
            'remember_likelihood': round(rmb, 4),
            'depth': depth,
            'ted_from_ltr': ted,
            'generation_order': order,
            'tree': tree_to_dict(tree, blocks, tokenizer, targets_main),
        })

    print(f"  [text={text_idx}, blk_sz={blk_sz}] computing LTR reference metrics...")
    ltr_tree_lik = compute_tree_likelihood(ltr_tree, G)
    ltr_rmb = compute_remember_likelihood(ltr_tree, blocks, targets, masked_input_ids)

    return {
        'text_idx': text_idx,
        'text': text,
        'blk_sz': blk_sz,
        'block_labels': block_labels,
        'num_blocks': num_blocks,
        'arborescences': arborescences,
        'ltr_tree_likelihood': round(ltr_tree_lik, 4),
        'ltr_remember_likelihood': round(ltr_rmb, 4),
    }


def main():
    results = []
    for text_idx in TEXT_INDICES:
        text = ds["train"][text_idx]["text"]
        print(f"\n=== Text {text_idx}: {text[:80]}... ===")
        for blk_sz in BLOCK_SIZES:
            data = analyze_one(text_idx, text, blk_sz)
            results.append(data)

    os.makedirs("static", exist_ok=True)
    with open("static/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to static/results.json")


if __name__ == "__main__":
    main()
