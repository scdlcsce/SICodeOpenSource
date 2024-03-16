import os
import sys

from tqdm import tqdm
import networkx as nx
import json
from gutils import *
import pickle as pkl
import time

MAXRE = 5000
sys.setrecursionlimit(MAXRE + 100)

corpus = []
        

def proc_json(pdata, astdata, fi, pdgdir):
    funcset = list()
    astdict = {}
    if not astdata == None:
        for lidx, l in enumerate(astdata):
            ltoken = l.split(':')
            if len(ltoken)<3:
                continue

    pbar = tqdm(total=len(pdata['functions']), desc='function in jsonfile')
    pdict = {}
    for pfunc in pdata['functions']:
        pdict[pfunc['function']] = pfunc['PDG']
    for fidx in range(len(pdata['functions'])):
        pbar.update(1)
        func = pdata['functions'][fidx]
        funcname = func['function']
        if funcname in funcset:
            continue
        else:
            funcset.append(funcname)
        funcpdg = pdict[funcname]
        if funcname == 'thunder_mmc_probe':
            print('debug')
        pdg = getgraph(funcname, funcpdg)  
        pdg = removeid(pdg) 
        pdg = normgraph(pdg) 
        pdg = removeassign(pdg)
        funccorpus = getcorpus(funcpdg, pdg)
        nlist = list(pdg.nodes())
        for n in nlist:
            if 'Method' in n:
                pdg.remove_node(n)
        nlist = list(pdg.nodes())
        count = 0
        for n in nlist:
            try:
                if pdg.nodes()[n]['type'] == 'api':
                    count += 1
            except:
                continue
        if count < 4:
            continue
        corpus.append(funccorpus)
        gp = pdgdir + str(fi) + '__' + funcname
        nx.write_gexf(pdg, gp +'.gexf')
        pkl.dump(pdg, open(gp + '.pkl','wb'))
        with open(gp + '.ast','w') as fp:
            try:
                fp.write(str(astdict[funcname]))
            except Exception:
                fp.write(str([]))
    pbar.close()



if __name__ == "__main__":
    print(time.time())
    pdgdir = '$GRAPHDIR'
    jsondir = '$JSONDIR'
    listdir = '$LISTDIR'
    c = 'find {} -name "*pdg.json" > {}'.format(jsondir, listdir)
    os.system(c)
    with open(listdir,'r') as fp:
        flist = fp.readlines()
    for f in flist:
        fi = flist.index(f)
        print('[' + str(fi) + '/' + str(len(flist) - 1) + ']: ' + f.strip())
        filepath = f.strip()
        try:
            with open(filepath, 'r') as fp:
                pdata = json.load(fp)
        except json.JSONDecodeError:
            continue
        try:
            with open(filepath[:-9]+'.c.ast') as fp:
                astdata = fp.readlines()
        except FileNotFoundError:
            astdata = None
        f = proc_json(pdata, astdata, fi, pdgdir)


    corpus = list(set(corpus))
    with open('$CORPUSDIR','w')as fp:
        fp.writelines('\n'.join(corpus)[1:])

    vocab = []
    for c in corpus:
        cset = c.replace('\n','').split(' ')
        vocab += cset
    vocab = ['[UNK]','[PAD]','[SEP]','[CLS]','[MASK]'] + list(set(vocab)-set(['']))
    vocab = '\n'.join(vocab)
    with open('$VOCABDIR','w') as fp2:
        fp2.write(vocab)
    print(time.time())