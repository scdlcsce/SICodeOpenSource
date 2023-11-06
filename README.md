# This is the data set of submission #149
The source code will be released upon publication.

## Data
### Training set and Testing set

sampledict.pkl is the backbone sample extracted from project OpenSSL.

testdict_28.pkl is the testset. The size of the target graph is set to 28.

### Method Encoding Dict

openssl_cluster_76.txt is the method encoding dict for project Openssl with 76 clustering centers.

vlc_cluster_76.txt is the method encoding dict for project VLC with 76 clustering centers.

kernel5_cluster_276.txt is the method encoding dict for project LinuxKernel-v5 with 276 clustering centers.

kernel6_cluster_276.txt is the method encoding dict for project LinuxKernel-v6 with 276 clustering centers.

### Embedding Vectors
(https://doi.org/10.5281/zenodo.8216758)

openssl-emb-0.pkl is the embedding vector for featured subgraphs of project Openssl.

vlc-emb-0.pkl is the embedding vector for featured subgraphs of project VLC.

kernel5emb.tgz is the embedding vector for featured subgraphs of project LinuxKernel-v5.

kernel6emb.tar.gz is the embedding vector folder for featured subgraphs of project LinuxKernel-v6.

---
## Results
### The confirmed bugs
The confirmed bugs are listed in file bugs.txt.
In order to obey the double-blind policy, we only publish the last six number of each commit id or confirmation message.