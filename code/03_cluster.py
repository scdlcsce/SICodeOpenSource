import numpy as np
from Cluster import kcluster
if __name__ == '__main__':
    ncluster = 76
    vocab = []
    with open('$VOCABDIR', 'r') as fp:
        lines = fp.readlines()
        for i in range(len(lines))[5:]:
            vocab.append(lines[i].strip())
    with open('$VECDIR', 'r') as fp:
        vocabvecft = eval(fp.readline().strip())[5:]
    assert(len(vocab) == len(vocabvecft))
    embarray = np.array(vocabvecft)
    clusterid, error, nfound, centers = kcluster (embarray, nclusters=ncluster, dist='u')
    clusterid = list(clusterid)
    dlabel = {}
    for i in range(len(vocab)):
        dlabel[vocab[i]] = clusterid[i]
    with open('$CLUSTERDIR', 'w')as fp:
        fp.write(str(dlabel))