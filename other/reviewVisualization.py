import json
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import random

def random_dark_color():
    max = 200
    r = random.randint(0, max)
    g = random.randint(0, max)
    b = random.randint(0, max)
    return f'#{r:02x}{g:02x}{b:02x}'

def visualize_papers(json_file):
    """
    Visualize paper citations as a directed graph with year on y-axis.
    Args:
        json_file: Path to JSON file or JSON data as string/list
    """
    random.seed(42)
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
    
    # Build paper index and add nodes
    paper_dict = {}
    for paper in papers:
        title = paper['title']
        paper_dict[title] = paper
        G.add_node(title, year=paper['year'], author=paper['author'])
    
    # Add edges (citations)
    edge_labels = {}
    for paper in papers:
        source = paper['title']
        for ref in paper.get('refers to', []):
            target = ref['title']
            G.add_edge(source, target)
            edge_labels[(source, target)] = ref.get('reason', '')
    
    # Create layout with year on y-axis (evenly spaced)
    pos = {}
    year_groups = defaultdict(list)
    
    # Group papers by year
    for node in G.nodes():
        year = G.nodes[node]['year']
        year_groups[year].append(node)
    
    # Position nodes with evenly spaced years
    sorted_years = sorted(year_groups.keys(), reverse=False)
    year_to_position = {year: i for i, year in enumerate(sorted_years)}
    
    for year in sorted_years:
        nodes_in_year = year_groups[year]
        num_nodes = len(nodes_in_year)
        y_pos = year_to_position[year]
        
        # Spread nodes horizontally
        for i, node in enumerate(nodes_in_year):
            x = (i - (num_nodes - 1) / 2) * 2
            pos[node] = (x, y_pos)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Draw edges with arrows
    edge_colors = [random_dark_color() for _ in G.edges()]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        #edge_color="#4A4A4A",
        edge_color=edge_colors,
        arrows=True,
        arrowsize=5,
        arrowstyle='-|>',
        width=1,
        style='dashed',
        connectionstyle='arc3,rad=0.1',
        node_size=20,
        min_source_margin=10,
        min_target_margin=10
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                          node_size=0, edgecolors='darkblue',
                          linewidths=1)
    
    # Draw labels with truncated title and author
    labels = {}
    for node in G.nodes():
        author = G.nodes[node]['author']
        year = G.nodes[node]['year']
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        # Break line at 30 characters, truncate at 60 with ...
        if len(node) <= 30:
            truncated_title = node
        elif len(node) <= 60:
            # Break into two lines at 30 characters
            truncated_title = node[:30] + '-\n' + node[30:]
        else:
            # Break at 30 and truncate second line at 60 total
            truncated_title = node[:30] + '-\n' + node[30:57] + '...'
        
        # Set background color based on out degree
        bg_color = "lightgreen" if out_degree > 0 else "lightgray"
        
        labels[node] = f"{truncated_title}\n{author} ({year}), Citations: {in_degree}"
        nx.draw_networkx_labels(G, pos, {node: labels[node]}, ax=ax, font_size=8,
                            font_weight='bold', 
                            bbox=dict(facecolor=bg_color, edgecolor="black", alpha=0.7, pad=3))
    
    # Draw edge labels (reasons for citations)
    """ nx.draw_networkx_edge_labels(
        G, pos, edge_labels, ax=ax,
        font_size=4,
        font_color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    ) """
    
    # Customize plot with evenly spaced year labels
    ax.set_title('Paper Citation Network', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Year', fontsize=12)
    
    # Set y-axis to show actual years at evenly spaced positions
    ax.set_yticks(range(len(sorted_years)))
    ax.set_yticklabels(sorted_years)
    
    ax.grid(True, alpha=0.3, axis='y')
    #ax.margins(0.2)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nGraph Statistics:")
    print(f"Total papers: {G.number_of_nodes()}")
    print(f"Total citations: {G.number_of_edges()}")
    print(f"Year range: {min(sorted_years)} - {max(sorted_years)}")

# Example usage
if __name__ == "__main__":
    # Load and visualize from your JSON file
    visualize_papers('papers.json')  # Replace with your actual filename