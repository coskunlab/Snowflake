import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, pos, color, ax=None):
    if ax == None:
        fig, ax = plt.subplots(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        ax.axis('equal')

    nx.draw_networkx(G, pos=pos, with_labels=False, edgecolors='k',
                     node_color=color, cmap="bwr", node_size=30, ax=ax, )
