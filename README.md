# abstractive_summarization

## Set up

Get dataset with:

```
wget http://sifaka.cs.uiuc.edu/ir/textdatabook/apnews.tar.gz
```

...then unzip tar file.

## Run search

```
pip3 install -r requirements.txt  # install metapy
python3 search_main.py
```

It will be slow on the first run because it has to build the index. After the
index is built, the subsequent runs will be much faster.
