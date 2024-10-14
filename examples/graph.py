import svs   # <-- pip install -U svs

from random import random
from itertools import combinations

import networkx as nx  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def build_graph(path: str) -> None:
    """
    This builds a small demo graph (without embeddings).

    In a _real_ app, you'd build a much larger graph, composed of documents
    with embeddings and metadata. This just shows you the interface!

    The beauty of SVS is that you can store your graph in the database
    along with your documents, embeddings, and metadata. Everything
    in one pace! Yay!

    Note: The SVS graph is composed of nodes (as documents) and edges (also
          as documents). That is, an edge is a document just like a node!
          This allows you to have text, embeddings, metadata in your edges!

    Remember: SVS is just the storage layer, offering you various optional
              features. It's up to you how to layer retrieval/agent algorithms
              on top of you SVS database(s).
    """
    kb = svs.KB(path, svs.make_mock_embeddings_func(), force_fresh_db=True)

    with kb.bulk_add_docs() as add_doc:
        doc_ids = [
            add_doc(
                text=char,
                no_embedding=True,
            )
            for char in 'abcdefghijk'
        ]
        edge_type_1 = add_doc(
            text='edge',
            no_embedding=True,
        )

    with kb.bulk_graph_update() as graph:
        for n1, n2 in combinations(doc_ids, r=2):
            if ((n1 % 3) == 1) and n2 - n1 <= 3:
                graph.add_directed_edge(n1, n2, edge_type_1, weight=random())

    kb.close()


def visualize_graph(path: str) -> None:
    kb = svs.KB(path)
    with kb.bulk_graph_update() as graph:
        G = graph.build_networkx_graph()
    kb.close()

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='twopi')
    nx.draw_networkx(G, pos=pos)
    weights = {
        k: f'{v:.1f}'
        for k, v in nx.get_edge_attributes(G, 'weight').items()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.savefig('./graph.png')


def main() -> None:
    path = './tempdb.sqlite'
    build_graph(path)
    visualize_graph(path)


if __name__ == '__main__':
    main()
