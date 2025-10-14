import json
import networkx as nx
from pyvis.network import Network

def visualize_papers_responsive(json_file, output_html="papers_network.html"):
    """
    Visualize paper citations as a responsive, interactive network graph.
    Works in Jupyter or exports to HTML with full viewport responsiveness.
    """
    # Load data
    if isinstance(json_file, str):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
        except FileNotFoundError:
            papers = json.loads(json_file)
    else:
        papers = json_file

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for paper in papers:
        title = paper['title']
        year = paper['year']
        author = paper['author']
        G.add_node(title, year=year, author=author)

    # Add edges (citations)
    for paper in papers:
        source = paper['title']
        for ref in paper.get('refers to', []):
            target = ref['title']
            reason = ref.get('reason', '')
            G.add_edge(source, target, title=reason)

    # Create PyVis network
    net = Network(
        height="90vh",   # viewport height
        width="100vw",   # viewport width
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222"
    )

    # Convert from NetworkX
    net.from_nx(G)

    # Enhance interactivity
    for node in net.nodes:
        title = node["id"]
        author = G.nodes[title].get("author", "Unknown")
        year = G.nodes[title].get("year", "N/A")
        node["title"] = f"<b>{title}</b><br>{author} ({year})"
        node["label"] = title[:25] + ("..." if len(title) > 25 else "")
        node["shape"] = "dot"
        node["size"] = 15

    for edge in net.edges:
        if "title" in edge:
            edge["title"] = edge["title"]
        edge["arrows"] = "to"
        edge["color"] = {"color": "#4A4A4A"}

    # Physics for smooth layout
    net.set_options("""
    {
      "physics": {
        "stabilization": false,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "springLength": 150,
          "springConstant": 0.05,
          "damping": 0.09
        }
      },
      "nodes": {
        "font": { "size": 14, "face": "arial" },
        "scaling": { "min": 10, "max": 30 }
      },
      "edges": {
        "smooth": {
          "type": "dynamic",
          "roundness": 0.2
        }
      }
    }
    """)


    # Show or save
    net.show(output_html, notebook=False)
    print(f"✅ Responsive network saved to: {output_html}")

    # Print stats
    print(f"\nGraph Statistics:")
    print(f"Total papers: {G.number_of_nodes()}")
    print(f"Total citations: {G.number_of_edges()}")

if __name__ == "__main__":
    visualize_papers_responsive('papers.json')
