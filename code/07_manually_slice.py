import networkx as nx
import pickle as pkl
rootpath = '$SEEDPATH'
graphpath = '$SEEDFILE'
g = pkl.load(open(rootpath+ graphpath, 'rb'))
inodes = ['Call@54','Call@5d','Call@80','Call@8b','Return@60','Return@90']
subg = g.subgraph(inodes)
seedg = nx.DiGraph()
for ni in subg.nodes():
    seedg.add_node(ni, stm = subg.nodes()[ni]['stm'], type = subg.nodes()[ni]['type'], line = subg.nodes()[ni]['line'])
seedg.add_edges_from(subg.edges())

pkl.dump(seedg, open(rootpath+'seed_'+graphpath, 'wb'))