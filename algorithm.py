from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp import sent_tokenize, word_tokenize
from pythainlp.corpus import thai_stopwords
import numpy as np
import pandas as pd
import glob
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import base64
import os


docs = pd.read_csv('datasets/Open source [WeVis-Promise Tracker] Data - promise.csv')
vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
doc_matrix = vectorizer.fit_transform(docs['promiseTitle'].values)
stopwords = list(thai_stopwords())

def get_title_of(row_id: int) -> str:
    select = docs['promiseTitle'][row_id]
    return select

def get_cluster_of(x:str, return_id=True) -> list:
    query = vectorizer.transform([x])
    cluster = []
    for i in range(len(docs)):
        if cosine_similarity(query, doc_matrix[i]) > 0.35:
            cluster.append(i) # use float to make it json serializable

    if not return_id:
        cluster = [get_title_of(i) for i in cluster]
    return sorted(cluster)

def get_cluster_groups(return_id=False) -> list:
    rid = 'id' if return_id else 'title'
    cache = glob.glob('cache/*.json')
    cache = [c for c in cache if rid in c]
    if len(cache) > 0:
        with open(cache[0], 'r') as f:
            clusters = json.load(f)
            return clusters
        
    clusters = []
    for i in tqdm(range(len(docs))):
        cluster = get_cluster_of(docs['promiseTitle'][i])
        if cluster not in clusters:
            cluster = sorted(cluster)
            clusters.append(cluster)

    # map cluster to title
    if not return_id:
        clusters = [[get_title_of(i) for i in cluster] for cluster in clusters]

    with open(f'cache/{rid}.json', 'w') as f:
        save_format = {
            'data': clusters,
        }
        json.dump(save_format, f)

    return clusters

def get_topic_of(group: list, bigram=False) -> str:
    group = [x[0] for x in group]
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 2) if bigram else (1, 1))
    vectorizer.fit_transform(group)
    exclusion = [' ', '  ', '\n', '\t', '\xa0', '\u200b', '\u200b', '\u200b']
    # get top 3 words from each cluster
    all_words = []
    for sentence in group:
        all_words.extend(word_tokenize(sentence))
    all_words = [w for w in all_words if w[0] not in stopwords]
    all_words = [w for w in all_words if len(w) > 1]
    all_words = [w for w in all_words if w[0] not in exclusion]

    # get top 3 words by tf-idf
    tfidf = vectorizer.transform(group)
    tfidf = tfidf.toarray()
    tfidf = tfidf.sum(axis=0)
    tfidf = tfidf.tolist()
    tfidf = dict(zip(vectorizer.get_feature_names_out(), tfidf))
    tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    tfidf = tfidf[:10]
    tfidf = [w[0] for w in tfidf if w[0] not in stopwords and w[0] not in exclusion]
    all_word_gram = []
    for ngram in tfidf:
        all_word_gram.extend(ngram.split(' '))
    all_word_gram = [w for w in all_word_gram if w not in stopwords]
    all_word_gram = [w for w in all_word_gram if len(w) > 1]
    all_word_gram = [w for w in all_word_gram if w not in exclusion]
    kw = list(set(all_word_gram))
    return kw

def get_cluster_kmean():
    kmeans = KMeans(n_clusters=24)
    clustered_docs = kmeans.fit_predict(doc_matrix)
    newdf = pd.DataFrame(columns=['data','party','label_clusters'],data=np.array([docs['promiseTitle'].values,docs['party'].values,clustered_docs]).T)
    all_df = newdf[newdf['label_clusters'] == 0]
    for i in range(1,24):
        all_df = all_df.append(newdf[newdf['label_clusters'] == i])
    mk_list = []
    for i in range(1,24):
        subdict = dict()
        subdict['group'] = get_topic_of(all_df[all_df['label_clusters'] == i][['data']].values.tolist())
        party = all_df[all_df['label_clusters'] == i][['party']].values
        title = all_df[all_df['label_clusters'] == i][['data']].values
        policy = []
        for i in range(len(party)):
            policy_dict = dict()
            policy_dict['party'] = party[i][0]
            policy_dict['title'] = title[i][0]
            policy.append(policy_dict)
        subdict['policy'] = policy
        mk_list.append(subdict)
    return mk_list        

def get_treemap():
    k_cluster = get_cluster_kmean()
    data = []
    for cluster in range(len(k_cluster)):
        name = ' '.join(k_cluster[cluster]['group'])
        size = len(k_cluster[cluster]['policy'])
        children = {
            'name': name,
            'value': size,
        }
        data.append(children)
    return data

def get_wordcloud():
    # read file from cache as base64
    cache = './cache/wordcloud.svg'
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            base64_encoded = base64.b64encode(f.read())
            return str(base64_encoded)

if __name__ == '__main__':
    # print(type(get_cluster_groups(return_id=False)))
    # print(get_cluster_of('สร้างโรงเรียนให้เด็กได้เรียนภาษาอังกฤษ', return_id=False))
    # for cluster_id, data in enumerate(get_cluster_groups(return_id=False)['data']):
    #     if len(data) > 1:
    #         print(' '.join(get_topic_of(data, bi)))
    #         print(data)
    #         print('-'*20)
    # print(get_topic_of())
    print(get_cluster_kmean(doc_matrix))
    # for cluster_id, data in enumerate(get_cluster_groups(return_id=False)['data']):
    #     if len(data) > 1:
    #         print(' '.join(get_topic_of(data, bigram=True)))
    #         print('-'*20)
    # # print