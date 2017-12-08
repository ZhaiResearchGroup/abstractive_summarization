# How to use

Generate dataset
(use -h for args)
```
python3 generate_reddit_data.py
```

Generate corpus
(use -h for args)
```
python3 generate_reddit_corpus.py
```

Generate model
```
cd ../model_builder/
python3 model_gen.py -md ../reddit_data/model/ -dd ../reddit_data/corpus/ -o ../reddit_data/model/reddit_data.model
```

Perform extraction
```
python3 extraction.py -q query -c reddit_data/reddit_documents/reddit_documents-config.toml -m reddit_data/model/reddit_data.model
```