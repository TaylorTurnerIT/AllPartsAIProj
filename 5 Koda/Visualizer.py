import json
import matplotlib
matplotlib.use("Agg")  # remove this line if running interactively
import matplotlib.pyplot as plt


def visualize_graph(graph_json_path, output_path, dpi=200):
    """
    Create a matplotlib visualization of the graph.

    Args:
        graph_json_path: Path to graph.json
        output_path: Path to save output image
        dpi: Image resolution (default: 200)
    """
    with open(graph_json_path) as f:
        graph = json.load(f)

    pos = {n["name"]: (n["center"]["X"], n["center"]["Y"]) for n in graph}
    edges = [(n["name"], dst) for n in graph for dst in n["connections"] if dst in pos]

    fig, ax = plt.subplots(figsize=(6, 6))

    xs = [pos[name][0] for name in pos]
    ys = [pos[name][1] for name in pos]
    ax.scatter(xs, ys)

    for name, (x, y) in pos.items():
        ax.text(x + 0.5, y + 0.5, name)

    for src, dst in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->"))

    margin = 5
    ax.set_xlim(min(xs)-margin, max(xs)+margin)
    ax.set_ylim(min(ys)-margin, max(ys)+margin)
    ax.invert_yaxis()     # match image coordinate system
    ax.set_title("Graph Visualizer Output")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved graph visualization to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Single Line Diagram (SLD) graph"
    )

    parser.add_argument(
        '--graph-json',
        default="graph.json",
        help='Path to input graph.json file (default: graph.json)'
    )

    parser.add_argument(
        '--output',
        default="graph_visualized_step4.png",
        help='Path to output visualization image (default: graph_visualized_step4.png)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='Output image resolution (default: 200)'
    )

    args = parser.parse_args()

    visualize_graph(args.graph_json, args.output, args.dpi)
