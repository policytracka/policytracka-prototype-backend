from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp import sent_tokenize, word_tokenize
from pythainlp.corpus import thai_stopwords
import pandas as pd
import glob
import json
from tqdm import tqdm
import numpy as np

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

if __name__ == '__main__':
    # print(type(get_cluster_groups(return_id=False)))
    # print(get_cluster_of('สร้างโรงเรียนให้เด็กได้เรียนภาษาอังกฤษ', return_id=False))
    # for cluster_id, data in enumerate(get_cluster_groups(return_id=False)['data']):
    #     if len(data) > 1:
    #         print(' '.join(get_topic_of(data, bigram=True)))
    #         print('-'*20)
    # # print