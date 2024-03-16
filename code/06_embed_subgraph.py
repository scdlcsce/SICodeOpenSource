import  torch
import argparse
import pickle
import  numpy as np
from config import parse_encoder
from modeldata import DiskDataSource
import models
import utils
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import pickle as pkl

input_dim = 100
wvec_dim = 76
opr_dim = 22

with open('$CLUSTERDIR', 'r') as fp:
    ldict = eval(fp.read())

def build_model(args):
    model = models.OrderEmbedder(args)
    model.to(utils.get_device())
    return model

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)

code2type = { 
    '<operator>.plus':                          0,
    '<operator>.addition':                      0,
    '<operator>.subtraction':                   1,
    '<operator>.multiplication':                2,
    '<operator>.division':                      3,
    '<operator>.modulo':                        4,
    '<operator>.or':                           5,
    '<operator>.and':                           6,
    '<operator>.not':                           7,
    '<operator>.shiftRight':                    8,
    '<operators>.LogicalShifRight':             8,
    '<operator>.arithmeticShiftRight':          8,
    '<operator>.arithmeticShiftLeft':           8,
    '<operator>.shiftLeft':                     8,
    '<operator>.equals':                        9,
    '<operator>.notEquals':                     10,
    '<operator>.lessEqualsThan':                11,
    '<operator>.lessThan':                      11,
    '<operator>.greaterEqualsThan':             12,
    '<operator>.greaterThan':                   12,
    '<operator>.logicalNot':                    13,
    '<operator>.logicalOr':                     14,
    '<operator>.logicalAnd':                    15,
    '<operator>.sizeOf':                        16,
    '<operator>.addressOf':                     17,
    # --- drop ---
    '<operator>.minus':                         18, 
    '<operator>.cast':                          19,
    '<operator>.indirectMemberAccess':          20,          
    '<operator>.computedMemberAccess':          20,
    '<operator>.indirection':                   20,
    '<operator>.memberAccess':                  20,
    '<operator>.assignment':                    21,
}

def g2feats(g):
    feat = np.zeros(input_dim)
    gnode = list(g.nodes())
    for n in gnode:
        callstm = g.nodes()[n]["stm"]
        calltype = g.nodes()[n]["type"]
        if calltype == 'api':
            label = ldict[callstm]
            gvec = label
        elif calltype == 'opr':
            label = code2type[callstm]
            gvec = wvec_dim + label
        elif calltype == 'return': 
            gvec = input_dim - 2
        elif calltype == 'block':
            gvec = input_dim - 1
        feat[gvec] += 1
        g.nodes()[n]['label'] = gvec
    for e in list(g.edges()):
        del g.edges()[e]['type']

    return feat, g


if __name__ == "__main__":
    parser = (argparse.ArgumentParser(description='Order embedding arguments'))
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args() 
    with open('$SUBGRAPHDIR', 'rb') as fp:
        [namelist, subgraphlist] = pickle.load(fp)
    featslist = []
    encg = []
    fnamelist = []
    for gidx, g in enumerate(subgraphlist):
        if len(list(g.edges())) == 0:
            continue
        r1, r2 = g2feats(g)
        featslist.append(r1)
        encg.append(r2)
        fnamelist.append(namelist[gidx])

    model_path = '$MODELDIR'
    model = models.OrderEmbedder(args)
    model.load_state_dict(torch.load(model_path, map_location="cuda:1"))
    model.to(torch.device("cuda:1"))
    model.eval()

    emb_S = []
    Sdata_source = DiskDataSource(encg)
    loaders = Sdata_source.gen_retrieval_loaders(len(encg), args.batch_size )
    for batch_i in tqdm(loaders):
        g = Sdata_source.gen_retrieval_batch(batch_i)
        with torch.no_grad():
            emb_g = model.emb_model(g)
        emb_S += emb_g
    emb_S = torch.stack(emb_S)
    emb_S = emb_S.cpu().detach()
    pkl.dump([fnamelist, emb_S, featslist], open('$EMBEDDINGDIR', 'wb'))