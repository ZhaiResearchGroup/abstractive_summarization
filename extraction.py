import numpy as np
import pandas as pd
import argparse
from searcher import *
import tokenizer
import graph_builder
import textrank
import DocumentGraph
import gensim


CORPUS_PATH = 'apnews/apnews.dat'
META_CONFIG_PATH = 'apnews/apnews-config.toml'
MODEL_PATH = 'model/apnews_sen_model.model'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", nargs='?', default='Airbus Subsidies', type=str, help='query')
    args = parser.parse_args()

    with open(CORPUS_PATH, 'r') as corpus:
        combined_document = corpus.read()
        corpus.seek(0)
        documents = corpus.readlines()

    combined_document = combined_document[:1000]
    documents = documents[:1695]

    print (len(documents))
    print (len(set(documents)))

    N_docs = len(documents)

    # run BM25
    searcher = Searcher(META_CONFIG_PATH)
    search_results = searcher.search(args.query, num_results=N_docs)

    dupe_dict = dict()
    for (doc_id,_) in search_results:
        if doc_id in dupe_dict:
            print ('oh no')
            return

        dupe_dict[doc_id] = True

    combined_document = searcher.get_stringified_list(search_results)


    # run textrank from law__--less
    tokenized_sentences = tokenizer.remove_stopwords_and_clean(combined_document)
    word_model = gensim.models.doc2vec.Doc2Vec.load(MODEL_PATH)
    graph_model = DocumentGraph.DocumentGraph(tokenized_sentences, word_model)
    M_adj = graph_model.similarity_matrix

    M_adj = M_adj / np.sum(M_adj, axis=1)
    eigen_vectors = np.array(textrank.textrank(M_adj, d=.85))
    scores = textrank.get_sentence_scores(tokenized_sentences, eigen_vectors)

    assert(len(combined_document) == len(scores))

    print (scores)

    all_scores = np.ndarray((scores.shape[0], 2))
    all_scores[:, 1] = scores
    all_scores[:, 0] = [result[1] for result in search_results]
    z_scores = (all_scores - np.mean(all_scores, axis=0)) / np.std(all_scores, axis=0)
    averaged_scores = np.mean(z_scores, axis=1)

    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    sorted_docs = [doc for (avg_score, doc) in sorted(zip(averaged_scores, combined_document), reverse=True)]

    print (sorted_docs)



if __name__ == '__main__':
    main()
