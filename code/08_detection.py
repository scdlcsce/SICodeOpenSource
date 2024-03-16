import  torch
import argparse
from config import parse_encoder
import models
from gutils import g2feats
import utils
from tqdm import tqdm
from tqdm import tqdm
import pickle as pkl
import numpy as np
input_dim = 100
wvec_dim = 76
opr_dim = 22
from modeldata import DiskDataSource
import time
 
g_seed = pkl.load(open('$SEEDDIR', 'rb'))
model_path = '$MODELDIR'
logpath = '$LOGDIR'
with open('$CLUSTERDIR', 'r') as fp:
    ldict = eval(fp.read())

def build_model(args):
    model = models.OrderEmbedder(args)
    model.to(utils.get_device())
    return model

if __name__ == "__main__":
    feat, g = g2feats(g_seed, input_dim, wvec_dim, ldict)
    sizeseed = sum(feat)
    parser = (argparse.ArgumentParser(description='Order embedding arguments'))
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    model = models.OrderEmbedder(args)
    model.to(utils.get_device())
    model.load_state_dict(torch.load(model_path))
    model.eval()

    Sdata_source = DiskDataSource([g_seed])
    loaders = Sdata_source.gen_retrieval_loaders(1, 1)
    for batch_i in tqdm(loaders):
        v = Sdata_source.gen_retrieval_batch(batch_i)
        with torch.no_grad():
            emb_v = model.emb_model(v)

    p_dict_all = {}
    for i in tqdm(range(0,int('($EMBFILRNUM'))):
        [namelist, emb_T, featslist] = pkl.load(open('$EMBFILE-{}.pkl'.format(i), 'rb'))
        featslist =np.asarray(featslist)
        sizetarget = torch.tensor(np.sum(featslist, axis= -1)).to(torch.device("cuda:1"))
        idxs = np.where(np.min(featslist - feat,axis = 1)>=0)[0]
        s_namelist = [namelist[i] for i in idxs.tolist()]
        s = emb_v.expand(emb_T.shape)
        raw_pred = model.predict(emb_T.to(utils.get_device()), s)
        pred = [0 if raw_pred[idx]> 1000 else 1 for idx in range(int(raw_pred.shape[0]))]
        p_dict = {namelist[idxp]:raw_pred[idxp].item() for idxp, p in enumerate(pred) if (p ==1 and namelist[idxp] in s_namelist) }
        for k in p_dict.keys():
            p_dict_all[k] = p_dict[k]
    
    p_dict = sorted(p_dict_all.items(), key=lambda d: d[1], reverse=False)[:100]

    p_func = []
    for p in p_dict:
        f = p[0].split('_Call@')[0]
        if not f in p_func:
            p_func.append(f)

    p_dict = [p[0]+','+str(p[1]) for p in p_dict]

    with open(logpath,'w')as fp:
        fp.write(str(p_dict)+'\n\n\n'+ str(p_func))
    print(time.time())
    