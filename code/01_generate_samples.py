import os
import networkx as nx
from tqdm import tqdm
import pickle as pkl
from operator import itemgetter
from itertools import groupby
import networkx as nx
import random as rd
from gutils import *

if __name__ == '__main__':
    pdgraphpath = '$GRAPHDIR'
    sourcelist = [s for s in list(os.walk(pdgraphpath))[0][2] if '.pkl' in s ]
    rd.shuffle(sourcelist)
    sampledict = {}
    subgs = []
    for hopsize in [1,2,3,4]:
        for fname in tqdm(sourcelist[:int(len(sourcelist)*0.8)]):
            pgraph = pkl.load(open(pdgraphpath+fname, 'rb'))
            for n in list(pgraph.nodes()):
                if 'MethodReturn' in n or 'Method' in n:
                    pgraph.remove_node(n)
            nlist = []
            for ni in list(pgraph.nodes()):
                if pgraph.nodes()[ni]['type'] == 'api':
                    nlist.append(ni)
            nlist = list(set(nlist))
            rd.shuffle(nlist)
            for n in nlist[:50]:
                g = g_sample(pgraph ,n, hopsize)
                if not g == None:
                    if not len(list(g.edges())) == 0:
                        g = gtostr(g, n)
                        subgs.append(str(g))
    rd.shuffle(subgs)
    subgs = list(set(subgs))
    subgs = [eval(s) for s in subgs]
    subgs.sort(key=itemgetter(0,1))
    subgs = groupby(subgs, itemgetter(0))
    samples = {}
    for k, v in subgs:
        samples[k] = list(v)[:1000]
    
    try:
        samples.pop(1)
    except:
        pass
    try:
        samples.pop(2)
    except:
        pass
    try:
        samples.pop(3)
    except:
        pass
    try:
        samples.pop(4)
    except:
        pass
    try:
        samples.pop(5)
    except:
        pass
    for k in samples.keys():
        samplelist = []
        for s in samples[k]:
            g = nx.Graph()
            t = eval(s[1])[0]
            e = eval(s[1])[1]
            for ni in range(s[0]):
                g.add_node('n_'+str(ni), type=t[ni])
            for ei in e:
                ei = eval(ei)
                g.add_edge('n_'+str(ei[0]), 'n_'+str(ei[1]))
            samplelist.append([s[0], t, e, g])
        sampledict[k] = samplelist
    klist = list(sampledict.keys())
    
    samples = {k:sampledict[k] for k in klist}

    with open('$TRAINSAMPLEDIR', 'wb') as fid:
        pkl.dump(samples, fid)


    sampledict = {}
    subgs = []
    for hopsize in [1,2,3,4]:
        for fname in tqdm(sourcelist[int(len(sourcelist)*0.8):]):
            pgraph = pkl.load(open(pdgraphpath+fname, 'rb'))
            nlist = []
            for ni in list(pgraph.nodes()):
                if pgraph.nodes()[ni]['type'] == 'api':
                    nlist.append(ni)
            
            nlist = list(set(nlist))
            rd.shuffle(nlist)

            for n in nlist[:50]:
                g = g_sample(pgraph ,n, hopsize)
                if not g == None:
                    if not len(list(g.edges())) == 0:
                        g = gtostr(g, n)
                        subgs.append(str(g))

    rd.shuffle(subgs)
    subgs = list(set(subgs))
    subgs = [eval(s) for s in subgs]
    subgs.sort(key=itemgetter(0,1))
    subgs = groupby(subgs, itemgetter(0))
    samples = {}
    for k, v in subgs:
        samples[k] = list(v)[:1000]
    
    try:
        samples.pop(1)
    except:
        pass
    try:
        samples.pop(2)
    except:
        pass
    try:
        samples.pop(3)
    except:
        pass
    try:
        samples.pop(4)
    except:
        pass
    try:
        samples.pop(5)
    except:
        pass

    for k in samples.keys():
        samplelist = []
        for s in samples[k]:
            g = nx.Graph()
            t = eval(s[1])[0]
            e = eval(s[1])[1]
            for ni in range(s[0]):
                g.add_node('n_'+str(ni), type=t[ni])
            for ei in e:
                ei = eval(ei)
                g.add_edge('n_'+str(ei[0]), 'n_'+str(ei[1]))
            samplelist.append([s[0], t, e, g])
        sampledict[k] = samplelist
    klist = list(sampledict.keys())
    
    samples = {k:sampledict[k] for k in klist}

    with open('$TESTSAMPLEDIR', 'wb') as fid:
        pkl.dump(samples, fid)
