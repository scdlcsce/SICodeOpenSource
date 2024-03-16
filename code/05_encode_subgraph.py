import argparse
from config import parse_encoder
from gutils import g_sample
import utils
from tqdm import tqdm
import os
from tqdm import tqdm
import pickle as pkl
import time

if __name__ == "__main__":
    print(time.time())
    parser = (argparse.ArgumentParser(description='Order embedding arguments'))
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()
    pdg_path = '$GRAPHDIR'
    funnames = [ p for p in list(os.walk(pdg_path))[0][2] if '.pkl' in p]
    namelist = []
    subgraphlist = []
    featslist = []
    lenlist = []
    
    for fname in tqdm(funnames):
        pgraph = pkl.load(open(pdg_path+fname, 'rb'))
        for n in list(pgraph.nodes()):
            if 'MethodReturn' in n or 'Method' in n:
                pgraph.remove_node(n)
        apilist = [ni for ni in list(pgraph.nodes()) if pgraph.nodes()[ni]['type'] == 'api']
        for n in apilist:
            subc = g_sample(pgraph ,n, 4)
            if subc == None:
                continue
            if len(list(subc.edges())) == 0:
                continue
            if len(list(subc.nodes())) < 3:
                continue
            subgraphlist.append(subc)
            namelist.append(fname[:-4]+'_'+n)
            lenlist.append(len(list(subc.nodes())))
    print(time.time())
    path = '$SUBGRAPHDIR'
    pkl.dump([namelist, subgraphlist], open(path, 'wb'))
