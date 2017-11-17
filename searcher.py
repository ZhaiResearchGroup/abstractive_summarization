import metapy

# code adapted from https://github.com/meta-toolkit/metapy/blob/master/tutorials/2-search-and-ir-eval.ipynb
class Searcher:

    def __init__(self, config_filename):
        # create the inverted index with the apnews data
        self.idx = metapy.index.make_inverted_index(config_filename)

        # create the ranker so we can perform BM25 search
        self.ranker = metapy.index.OkapiBM25()

    def search(self, raw_query, num_results=10):
        '''
            Searches an index of documents given a query.

            Returns a ranked list of the top results, where each result is
            in the format: (doc_id, score).
        '''
        # create a search query
        query = metapy.index.Document()
        query.content(raw_query)
        
        # search the index
        search_results = self.ranker.score(self.idx, query, num_results=num_results)
        return search_results

    def stringify_search_results(self, search_results):
        '''
            "Stringifies" the search results by returning a string of the first
            sentence for each document.
        '''
        result = ""
        for i, (doc_id, score) in enumerate(search_results):
            doc_content = self.idx.metadata(doc_id).get('content')
            first_sentence = self._get_first_sentence(doc_content)

            result += "{}. {}\n".format(i + 1, first_sentence)

            if i != len(search_results)-1:
                result += "\n"

        return result

    def _get_first_sentence(self, raw_doc_content, sentence_delimiter="   "):
        '''
            Gets the first sentence of a document
        '''

        # get rid of the three spaces at the very start
        doc_content = raw_doc_content[3:]

        first_sentence_end_idx = doc_content.find(sentence_delimiter)
        return doc_content[:first_sentence_end_idx]
