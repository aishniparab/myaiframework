import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_graph(h, color, epoch=None, loss=None, title='', figsize=(15, 15)):
    '''
    helper function for visualization
    ref: https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8#scrollTo=zF5bw3m9UrMy
    '''
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(title))
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        color = color.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set3")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set3", edge_color=(0,0,0,0.05))
    #plt.savefig('bongard_sample_as_graph.png', dpi=500, bbox_inches = 'tight')
    plt.show()

def visualize_tsne(h, color, epoch=None, loss=None, title='', figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(title))
    if torch.is_tensor(h):
      h = h.detach().cpu().numpy()
      color = color.detach().cpu().numpy()
      z = TSNE(n_components=2).fit_transform(h)
      plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
      if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
