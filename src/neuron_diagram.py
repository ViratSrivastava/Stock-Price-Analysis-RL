# neuron_diagram.py
# Render a neuron-level MLP diagram with Graphviz, sampling large layers for readability.

from graphviz import Digraph
import math
from typing import List, Optional

def sample_indices(n: int, k: int) -> List[int]:
    """Evenly sample k indices from range(n), preserving first/last for edges."""
    if k >= n:
        return list(range(n))
    # Even spacing across [0, n-1]
    return sorted(set([0, n-1] + [round(i*(n-1)/(k-1)) for i in range(k)]))[:k]

def format_layer_label(name: str, size: int, extras: Optional[str] = None) -> str:
    lab = f"{name} ({size})"
    if extras:
        lab += f"\\n{extras}"
    return lab

def draw_mlp_neurons(
    layer_sizes: List[int],
    layer_names: Optional[List[str]] = None,
    layer_extras: Optional[List[Optional[str]]] = None,
    max_neurons_per_layer: int = 32,
    filename: str = "dqn_neuron_diagram",
    engine: str = "dot",
    rankdir: str = "LR",
    node_size: str = "0.3",   # inches
    node_color: str = "lightgray",
    edge_color: str = "gray50",
    dpi: int = 150
):
    """
    layer_sizes: e.g., [state_dim, 1024, 512, 256, 128, 64, action_dim]
    layer_names: e.g., ["Input", "Dense1", "Dense2", "Dense3", "Dense4", "Dense5", "Output"]
    layer_extras: e.g., [None, "LayerNorm+ReLU+Dropout(0.2)", "LayerNorm+ReLU+Dropout(0.15)", ... , "Linear"]
    """
    assert len(layer_sizes) >= 2, "Need at least input and output layers"
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(len(layer_sizes))]
    if layer_extras is None:
        layer_extras = [None] * len(layer_sizes)
    assert len(layer_names) == len(layer_sizes)
    assert len(layer_extras) == len(layer_sizes)

    # Prepare neuron indices per layer (sampling if large)
    layer_index_maps = []
    for sz in layer_sizes:
        idxs = sample_indices(sz, max_neurons_per_layer)
        layer_index_maps.append(idxs)

    g = Digraph(filename=filename, format="png", engine=engine)
    g.attr(rankdir=rankdir)
    g.attr("graph", dpi=str(dpi), splines="spline", pad="0.2", nodesep="0.2", ranksep="1.2")
    g.attr("node", shape="circle", width=node_size, height=node_size, fixedsize="true",
           style="filled", fillcolor=node_color, color="black", penwidth="1")
    g.attr("edge", color=edge_color, arrowsize="0.6", penwidth="0.8")

    # Create subgraphs per layer (cluster to keep nodes grouped and aligned)
    layer_node_ids: List[List[str]] = []
    for li, (name, sz, extras, idxs) in enumerate(zip(layer_names, layer_sizes, layer_extras, layer_index_maps)):
        with g.subgraph(name=f"cluster_{li}") as s:
            s.attr(label=format_layer_label(name, sz, extras), labelloc="t", labeljust="c",
                   color="gray70", style="rounded")
            s.attr(rank="same")
            nodes_this_layer = []
            for j, orig_idx in enumerate(idxs):
                nid = f"L{li}_N{orig_idx}"
                s.node(nid, label="")
                nodes_this_layer.append(nid)
            layer_node_ids.append(nodes_this_layer)

            # Invisible chain to enforce vertical ordering within the layer (keeps neat stacks)
            for a, b in zip(nodes_this_layer, nodes_this_layer[1:]):
                s.edge(a, b, style="invis")

    # Connect layers (fully connect sampled neurons between adjacent layers)
    for li in range(len(layer_sizes) - 1):
        left_nodes = layer_node_ids[li]
        right_nodes = layer_node_ids[li + 1]
        # Dense bipartite edges can be heavy; consider downsampling further if needed
        for ln in left_nodes:
            for rn in right_nodes:
                g.edge(ln, rn)

    # Render to file
    outpath = g.render(cleanup=True)
    print(f"Saved diagram to: {outpath} (and source: {filename}.gv)")

if __name__ == "__main__":
    # Example for your DQNAgent MLP:
    # Replace these with real values:
    state_dim = 59  # e.g., your actual state_dim
    action_dim = 3   # e.g., your actual action_dim

    layer_sizes = [state_dim, 1024, 512, 256, 128, 64, action_dim]
    layer_names = [
        "Input",
        "Dense 1024",
        "Dense 512",
        "Dense 256",
        "Dense 128",
        "Dense 64",
        "Output"
    ]
    # Extras reflect your architecture blocks
    layer_extras = [
        None,
        "LayerNorm+ReLU+Dropout(0.2)",
        "LayerNorm+ReLU+Dropout(0.15)",
        "LayerNorm+ReLU+Dropout(0.1)",
        "LayerNorm+ReLU+Dropout(0.05)",
        "ReLU",
        "Linear (Q-values)"
    ]

    draw_mlp_neurons(
        layer_sizes=layer_sizes,
        layer_names=layer_names,
        layer_extras=layer_extras,
        max_neurons_per_layer=32,   # try 24 or 16 for lighter graphs
        filename="dqn_neuron_diagram",
        rankdir="LR",
        node_size="0.28",
        dpi=150
    )
