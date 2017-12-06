# abstractive_summarization

## Set up for dataset and model generation
1. Make sure each file of the apnews dataset is in the apnews_source_dataset directory.
2. Run python3 corpus_builder.py from within the corpus_building directory.

Now all of the sentences have been extracted from the apnews dataset and exist in their corresponding
files in the 'corpus/' directory.

Once the corpus has been generated, to generate the model:

3. Run python3 model_gen.py from within the model_builder directory.
4. At any point, if you keyboard interrupt to stop training, it will finish the current chunk of sentences and save the model to the 'model/' directory. It will also save a .txt file holding the current index of the documents being trained on to begin at when the model is next trained.

If there is a model in the 'model/' directory, the program will load the model from that folder and continue training that model.

When training finishes, the model will be saved to the 'model'/ directory, and the .txt file holding the current index will be removed, signifying that the model has been trained on the entire dataset.

## Search Set up

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
