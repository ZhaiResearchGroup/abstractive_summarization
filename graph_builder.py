
###########################################################################
#
#  law__--less/summarizer/graph_builder.py
#
###########################################################################

import numpy as np

'''
num_sentences_in_doc: An integer representing the number of sentences in the doc
returns             : A zero initialized numpy matrix
'''
def initialize_adj_matrix(num_sentences_in_doc):
    return np.zeros( ( num_sentences_in_doc , num_sentences_in_doc ) , dtype=np.int)

'''
s_array_len     : Number of sentences in the doc
unique_terms_len: Number of unique terms in the doc
returns         : A zero initialized numpy matrix
'''
def initialize_doc_matrix(s_array_len, unique_terms_len):
    return np.zeros( ( s_array_len , unique_terms_len ) , dtype=np.int)

'''
doc_s_matrix: A list of every sentence and an accompanying bitvector for the words in it
adj_matrix  : A zero initialized numpy array.
returns     : A completed adj matrix
'''
def build_adj_matrix(adj_matrix, doc_s_matrix):
    # for s1_index in range( len( doc_s_matrix ) ):
    #     for s2_index in range( len( doc_s_matrix ) ):
    #         adj_matrix[s1_index][s2_index] = compute_similarity( doc_s_matrix[s1_index], doc_s_matrix[s2_index])

    return np.dot(doc_s_matrix, doc_s_matrix.T)

'''
doc_s_matrix: A zero initialized numpy array
unique_terms: A list of all uniqe terms in doc
s_array     : A list of all sentence in doc where each sentence is a list of words
returns     : A completed document matrix
'''
def build_doc_matrix(s_array, doc_s_matrix, unique_terms):
    for sentence in range( len( doc_s_matrix ) ):
        for uq_index in range( len( unique_terms ) ):
            if unique_terms[uq_index] in s_array[sentence]:
                doc_s_matrix[sentence][uq_index] = 1
    return doc_s_matrix

'''
sentence1: A numpy array bit vector of all terms in doc with 1 for all terms in sentence1
sentence2: A numpy array bit vector of all terms in doc with 1 for all terms in sentence1
returns  : An integer similarity score using Textrank similarity
'''
def compute_similarity(sentence1, sentence2):
    # similarity_score = 0
    # for word_index in range( len( sentence1 ) ):
    #     if sentence1[ word_index ] and sentence2[ word_index ]:
    #         similarity_score +=1
    # return similarity_score

    return np.dot(sentence1, sentence2)

'''
adj_matrix  : A matrix where each sentence is adjacent by some weight
returns     : A list of all unique terms in the document
'''
def find_unique_terms(s_array):
    unique_terms = set()
    for sentence in s_array:
        # print (sentence)
        for term in sentence:
            # print (term)
            if term not in unique_terms:
                unique_terms.add(term)
    return list(unique_terms)

###########################This is effectively the main#########################
'''
s_array: A list of sentences where each sentence is a list of terms
returns: An adj matrix where the adj score is the textrank similarity score
'''
def create_sentence_adj_matrix(s_array):
    unique_terms = find_unique_terms(s_array)
    doc_s_matrix = initialize_doc_matrix( len(s_array), len(unique_terms))
    doc_s_matrix = build_doc_matrix(s_array, doc_s_matrix, unique_terms)
    adj_matrix = initialize_adj_matrix( len( doc_s_matrix ))
    adj_matrix = build_adj_matrix(adj_matrix, doc_s_matrix)
    return adj_matrix
