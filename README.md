## SEA-DialogGen

Code for paper "Search-Engine-augmented Dialogue Response Generation with Cheaply Supervised Query Production" (AI)

### Dependency

* coreference resolution provide by [AllenNLP ](https://demo.allennlp.org/coreference-resolution)
* pretrained language models by [Huggingface](https://huggingface.co/transformers/)
* test metrics by [Parlai](https://parl.ai/)

### Data Preparation

Please prepare dataset for query production following the scripts under `query-producer/data` indexing from 1-8. After a query producer is trained and produces the queries, use the 9-th script to get dataset for response generation.  Please be careful to set right file paths in each scripts.

### Query Production

* first prepare dataset for directly training using `prepare_data.py` 
* run the scripts under `query-producer/script` to train or test producers
* both extraction-based and generation-based are provided, but be careful to retrieve articles again when using generation-based, as the generated queries may not appear in the prepared query candidates by TagMe.

### Response Generation

* prepare dataset for directly training using `prepare_data.py` 
* run the scripts under `rank-gen/script` or `merge-gen/script` to train or test models

#### Potential Bug
* We notice a version mismatch problem when using latest version of transformers package. This bug influences generation models using BART because latest BART changes start token when generation. And, this bug can be solved by add ``decoder_start_token_id=0'' command for ``generate'' function (e.g., line 183 in Merge-gen/main.py).
