import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

path = "cleaned_data.csv"
data = pd.read_csv(path, sep=",", encoding='utf-8')
hub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(hub_url)

print('Embedding...')
for k,g in data.groupby(np.arange(len(data))//10):
    if k == 0:
        embeddings = embed(g['cleaned'])
    else:
        embeddings_new = embed(g['cleaned'])
        embeddings_new
        embeddings = tf.concat(values=[embeddings,embeddings_new],axis = 0)
    print(k)
print("The embeddings vector is of fixed length {}".format(embeddings.shape[1]))

np.save('embeddings_transformer.npy', embeddings, allow_pickle=True, fix_imports=True)