import pickle as pkl

import argparse
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import modeldata
import models
import utils
from utils import DupStdoutFileManager

from config import parse_encoder

seed = 2
np.random.seed(seed)
torch.random.manual_seed(seed)

def build_model(args):
    model = models.OrderEmbedder(args)
    model.to(utils.get_device())
    return model


def train_loop(args):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))

    record_keys = ["conv_type", "n_layers", "hidden_dim", "margin", "batch_size", "n_batches"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    print(args_str)
    model = build_model(args)
    optimizer = optim.Adam(model.emb_model.parameters(), lr=args.lr )
    batch_n = 0
    datasource = pkl.load(open('$TRAINSAMPLEDIR', 'rb'))
    data_source = modeldata.DiskDataSource(datasource)
    
    model.train()

    for nodenum in range(10, 70):
        if not nodenum in datasource:
            continue
        loaders = data_source.gen_data_loaders(args.n_batches // 60, args.batch_size)
        trainloss1, trainloss2, avg_p, avg_n = 0, 0, 0, 0
        pbar = tqdm(loaders)
        for batch_i in pbar:
            pbar.set_description("Loss1: {:.7f}. Loss2: {:.7f}. P Avg.: {:.4f}. N Avg.: {:.4f}".format(trainloss1, trainloss2, avg_p, avg_n))
            optimizer.zero_grad()
            anc, pos, neg1, neg2, samplenum = data_source.gen_batch(batch_i, nodenum)
            
            emb_anc = model.emb_model(anc)
            emb_pos, emb_neg1 = [], []

            for s in range(samplenum):
                emb_pos.append(model.emb_model(pos[s]))
                emb_neg1.append(model.emb_model(neg1[s]))
            
            loss1 = 0
            loss2 = 0

            for s in range(samplenum):
                lossi1 = torch.sum(torch.max(torch.zeros_like(emb_anc, device=emb_anc.device), 0.001*(s+1) + emb_anc - emb_pos[s]))
                lossi2 = torch.sum(torch.max(torch.zeros((args.batch_size, 1), device=emb_anc.device),  args.margin*(samplenum - s) - model.predict(emb_neg1[s], emb_anc)))
                loss1 += lossi1
                loss2 += lossi2
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                predp1 = model.predict(emb_pos[-1], emb_anc)
                predn1 = model.predict(emb_neg1[-1], emb_anc)
            trainloss1 = loss1.item()
            trainloss2 = loss2.item()
            avg_p = (torch.sum(predp1) / batch_i).item()
            avg_n = (torch.sum(predn1) / batch_i).item()

            batch_n += 1
            if batch_n % 1000 == 0:
                print('save')
                torch.save(model.state_dict(), '$CKPTDIR/ckpt/params_{:03}_{}n_1024.pt'.format(int(batch_n/1000), nodenum))


if __name__ == '__main__':
    parser = (argparse.ArgumentParser(description='Order embedding arguments'))
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager('$LOGDIR/train-' + now_time + '.log') as _:
        train_loop(args)
