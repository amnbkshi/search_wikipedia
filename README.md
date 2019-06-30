# search_wikipedia
The project uses a topic modelling approach to search a phrase on wikipedia and get the best sentences related to the query. Non - Negative Matrix Factorization (NMF) is used to to split the document-word matrix (A) into document-topic matrix (W) and topic-word matrix(H). Best sentences are then filtered out using scores from W. The final output is shown in a chronological order.

To extract the data python's wikipedia module is used. nltk & sklearn are used for text pre-processing and the model's algorithm.
