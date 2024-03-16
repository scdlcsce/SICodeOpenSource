from gensim.models import word2vec
from gensim.models import FastText
import time

if __name__ == "__main__":
    print(time.time())
    with open('$VOCABDIR')as fp:
        vocab = fp.readlines()
    sentences = word2vec.Text8Corpus('$CORPUSDIR')
    model = FastText(sentences, size=64, window=3, min_count=1)
    emb = []
    for i in vocab:
        i = i.strip()
        try:
            e = model[i].tolist()
        except Exception:
            e = None
        emb.append(e)
    with open('$VECDIR','w') as fp:
            fp.write(str(emb))

    