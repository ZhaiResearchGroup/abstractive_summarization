import numpy as np
import pandas as pd
import argparse
from searcher import *
import tokenizer
import textrank
import DocumentGraph
import gensim


META_CONFIG_PATH = 'apnews/apnews-config.toml'
MODEL_PATH = 'model/apnews_sen_model.model'
N_DOCS = 1695


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", nargs='?', default='Airbus Subsidies', type=str, help='query')
    args = parser.parse_args()

    # run BM25
    searcher = Searcher(META_CONFIG_PATH)
    search_results = searcher.search(args.query, num_results=N_DOCS)

    # sanity check for no duplicate doc_ids
    id_set = set()
    for (doc_id,_) in search_results:
        assert doc_id not in id_set, "duplicate search result doc_id: %r" % doc_id
        id_set.add(doc_id)

    search_sen = searcher.get_stringified_list(search_results)

    # get M_adj matrix using doc2vec
    tokenized_sentences = tokenizer.remove_stopwords_and_clean(search_sen)
    word_model = gensim.models.doc2vec.Doc2Vec.load(MODEL_PATH)
    graph_model = DocumentGraph.DocumentGraph(tokenized_sentences, word_model)
    M_adj = graph_model.similarity_matrix

    # run textrank
    M_adj = M_adj / np.sum(M_adj, axis=1)
    eigen_vectors = np.array(textrank.textrank(M_adj, d=.85))
    scores = textrank.get_sentence_scores(tokenized_sentences, eigen_vectors)

    # sanity check
    assert len(search_sen) == len(scores), "len(search_sen) != len(scores)"

    # average scores
    all_scores = np.ndarray((scores.shape[0], 2))
    all_scores[:, 1] = scores
    all_scores[:, 0] = [result[1] for result in search_results]
    z_scores = (all_scores - np.mean(all_scores, axis=0)) / np.std(all_scores, axis=0)
    averaged_scores = np.mean(z_scores, axis=1)

    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    sorted_docs = [doc for (avg_score, doc) in sorted(zip(averaged_scores, search_sen), reverse=True)]

    print (sorted_docs)



if __name__ == '__main__':
    main()
