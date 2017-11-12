from searcher import Searcher

def main():
    searcher = Searcher('apnews-config.toml')
    search_results = searcher.search('Airbus Subsidies')
    search_results_string = searcher.stringify_search_results(search_results)
    print(search_results_string)

if __name__ == '__main__':
    main()
