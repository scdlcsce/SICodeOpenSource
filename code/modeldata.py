import random as rd
import networkx as nx

import utils

rd.seed(2)
inputdim = 100
wvec_dim = 76
# opr_dim = 22


class DiskDataSource:
    def __init__(self, dataset):
        self.dataset = dataset

    def gen_data_loaders(self, batch_num, batch_size):
        loaders = [batch_size]* batch_num
        return loaders

    def gen_batch(self, batch_i, nodenum):
        batch_size = batch_i
        trainsets = self.dataset[nodenum]
        trainset = trainsets
        anclist, poslists, neglists = [], [], []
        ancnum = rd.choice(list(range(4, min(10, nodenum - 4))))
        idxlist = list(range(ancnum + 1, nodenum))
        idxlist = sorted(rd.sample(idxlist, 4)+[nodenum])
        i = 0
        while i < batch_size:
            poslist = []
            sample = rd.choice(trainset)
            types = sample[1]
            edge = [eval(e) for e in sample[2]]
            ns = []
            for nidx in range(nodenum):
                if types[nidx] == 'api':
                    ns.append(rd.choice(list(range(wvec_dim))))
                else:
                    ns.append(rd.choice(list(range(wvec_dim, inputdim))))
            
            alist = list(range(0,ancnum))
            e = [i for i in edge if i[0] < ancnum and i[1] < ancnum]
            if len(e) == 0:
                continue
            ga = nx.Graph()
            for ni in alist:
                ga.add_node('n_'+str(ni), label = ns[ni])
            for ei in e:
                ga.add_edge('n_'+str(ei[0]), 'n_'+str(ei[1]))

            for n in idxlist:
                nlist = list(range(0,n))
                e = [i for i in edge if i[0] < n and i[1] < n]
                gi = nx.Graph()
                for ni in nlist:
                    gi.add_node('n_'+str(ni), label = ns[ni])
                for ei in e:
                    gi.add_edge('n_'+str(ei[0]), 'n_'+str(ei[1]))
                poslist.append(gi)

            ancapi = [ni for ni in ns[:ancnum] if ni < wvec_dim]
            ancopr = [ni for ni in ns[:ancnum] if (ni >= wvec_dim)]

            neglist = []

            sample = rd.choice(trainset)
            types = sample[1]
            edge = [eval(e) for e in sample[2]]
            
            napi = [rd.choice(list(range(wvec_dim))) for nidx in range(0,len(types)) if types[nidx] == 'api' ]
            nopr = [rd.choice(list(range(wvec_dim, inputdim))) for nidx in range(0,len(types)) if types[nidx] is not 'api' ]

            napi = ancapi + napi[len(ancapi):]
            rd.shuffle(napi)
            nopr = ancopr + nopr[len(ancopr):]
            rd.shuffle(nopr)

            napicount = 0
            noprcount = 0
            nns = []
            for idx in range(len(types)):
                t = types[idx]
                if t == 'api':
                    nns.append(napi[napicount])
                    napicount += 1
                else:
                    nns.append(nopr[noprcount])
                    noprcount += 1
                

            for n in idxlist:
                nlist = list(range(0,n))
                e = [i for i in edge if i[0] < n and i[1] < n]
                gi = nx.Graph()
                for ni in nlist:
                    gi.add_node('n_'+str(ni), label = nns[ni])
                for ei in e:
                    gi.add_edge('n_'+str(ei[0]), 'n_'+str(ei[1]))
                neglist.append(gi)
            if len(list(poslist[0].edges())) == 0 or len(list(neglist[0].edges())) == 0:
                continue
            anclist.append(ga)
            poslists.append(poslist)
            neglists.append(neglist)
            i += 1

        pos = []
        n = []
        anc = utils.batch_nx_graphs(anclist)
        for i in range(5):
            plist = [poslists[j][i] for j in range(batch_i)]
            nlist = [neglists[j][i] for j in range(batch_i)]
            pos.append(utils.batch_nx_graphs(plist)) 
            n.append(utils.batch_nx_graphs(nlist)) 

        return anc, pos, n, None, 5
    
    def gen_retrieval_loaders(self, size, batch_size):
        loaders = [list(range(i, min(i+batch_size, size))) for i in list(range(0,size, batch_size))]
        return loaders
    def gen_retrieval_batch(self, batch_i):
        graphs = self.dataset
        g = [] 
        for i in  batch_i :
            sample = graphs[i]
            g.append(sample)
        g = utils.batch_nx_graphs(g)
        return g

class TestDataSource:

    def __init__(self, dataset):
        self.dataset = dataset

    def gen_data_loaders(self, batch_num, batch_size):
        loaders = [batch_size]* batch_num
        return loaders

    def gen_batch(self, batch_i, nodenum):
        batch_size = batch_i
        trainset = self.dataset[nodenum]

        pos_s_list, pos_l_list, neg_s_list, neg_l_list = [], [], [], []

        i = 0
        while i < batch_size:
            ancnum = rd.choice(list(range(3, min(10, nodenum - 1))))
            sample = rd.choice(trainset)

            types = sample[1]
            edge = [eval(e) for e in sample[2]]
            ns = []
            for nidx in range(nodenum):
                if types[nidx] == 'api':
                    ns.append(rd.choice(list(range(wvec_dim))))
                else:
                    ns.append(rd.choice(list(range(wvec_dim, inputdim))))

            
            gpl = nx.Graph()
            for ni in range(0,nodenum):
                gpl.add_node('n_'+str(ni), label = ns[ni])
            for ei in edge:
                gpl.add_edge('n_'+str(ei[0]), 'n_'+str(ei[1]))

            alist = list(range(0,ancnum))
            e = [i for i in edge if i[0] < ancnum and i[1] < ancnum]
            if len(e) == 0:
                continue
            gps = nx.Graph()
            for ni in alist:
                gps.add_node('n_'+str(ni), label = ns[ni])
            for ei in e:
                gps.add_edge('n_'+str(ei[0]), 'n_'+str(ei[1]))


            ancapi = [ni for ni in ns[:ancnum] if ni < wvec_dim]
            ancopr = [ni for ni in ns[:ancnum] if (ni >= wvec_dim)]

            sample = rd.choice(trainset)
            types = sample[1]
            edge = [eval(e) for e in sample[2]]
            
            napi = [rd.choice(list(range(wvec_dim))) for nidx in range(0,len(types)) if types[nidx] == 'api' ]
            nopr = [rd.choice(list(range(wvec_dim, inputdim))) for nidx in range(0,len(types)) if types[nidx] is not 'api' ]

            napi = ancapi + napi[len(ancapi):]
            rd.shuffle(napi)
            nopr = ancopr + nopr[len(ancopr):]
            rd.shuffle(nopr)

            napicount = 0
            noprcount = 0
            nns = []
            for idx in range(len(types)):
                t = types[idx]
                if t == 'api':
                    nns.append(napi[napicount])
                    napicount += 1
                else:
                    nns.append(nopr[noprcount])
                    noprcount += 1

            nlist = list(range(0,nodenum))
            gnl = nx.Graph()
            for ni in nlist:
                gnl.add_node('n_'+str(ni), label = nns[ni])
            for ei in e:
                gnl.add_edge('n_'+str(ei[0]), 'n_'+str(ei[1]))

            
            pos_s_list.append(gps)
            pos_l_list.append(gpl)
            neg_l_list.append(gnl)
            i += 1

        pos_s = utils.batch_nx_graphs(pos_s_list)
        neg_s = pos_s
        pos_l = utils.batch_nx_graphs(pos_l_list)
        neg_l = utils.batch_nx_graphs(neg_l_list)
        return pos_l, pos_s, neg_l, neg_s
