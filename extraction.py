import numpy as np
import pandas as pd
import argparse
from DataGen import *
from searcher import *
import tokenizer
import graph_builder
import textrank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", nargs='?', default='Airbus Subsidies', type=str, help='query')
    args = parser.parse_args()

    corpus_path = 'apnews/apnews_sen.dat'

    with open(corpus_path, 'r') as corpus:
        combined_document = corpus.read()
        documents = corpus.readlines()
        corpus.close()

    combined_document = combined_document[:1000]

    N_docs = len(documents)

    # run BM25
    # searcher = Searcher('apnews-config.toml')
    # search_results = searcher.search(args.query, num_results=N_docs)
    #
    # print (args.query, search_results)
    #
    # search_results = dict(search_results)

    # run textrank from law__--less
    tokenized_sentences = tokenizer.remove_stopwords_and_clean(tokenizer.tokenize_text(combined_document))
    M_adj = graph_builder.create_sentence_adj_matrix(tokenized_sentences).astype(float)
    M_adj = M_adj / np.sum(M_adj, axis=1)
    eigen_vectors = np.array(textrank.textrank(M_adj, d=.85))
    scores = textrank.get_sentence_scores(tokenized_sentences, eigen_vectors)

    # combine scores
    for i in range(scores.shape[0]):
        print (cores[i])




if __name__ == '__main__':
    main()
