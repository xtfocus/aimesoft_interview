---
id: 1708742098-FITW
aliases:
  - BERT
  - Bert
tags: []
---

# BERT

aka Bidirectional Encoder Representations from Transformers is a groundbreaking language model compared to its predecessors like recurrent neural networks (RNNs), word2vec (CBOW, Skip-gram), and tf-idf.

BERT uses Encoder in the [[1708742710-KDYY|transformers architecture]], introduced by the "Attention Is All You Need" paper. BERT is Encoder-only.

The output of BERT are embeddings, not predicted next words like GPT's.

## BERT vs CBOW

BERT is a masked language model (MLM), meaning the training objective of BERT it to predict the masked token in a sequence, somewhat similar to Continuous Bag of Words (CBOW). Both use *Self-Supervised Learning*, a fancy way to say that no labelling needed for training (thank to the masking trick).

What makes BERT different from CBOW is:
- the encoder architecture (which allows bidirectionality), 
- not using windows (so broader context compared to CBOW).
- BERT was also trained on *next sentence prediction* task (NSP): During pre-training, pairs of sentences are fed into the model, and the model is tasked with predicting whether the second sentence in the pair follows the first sentence in the original text. This objective helps BERT learn relationships between sentences and understand discourse-level context.

## Variants

1. The OG BERT: 
- wordpiece [[1708763669-GKCN|tokenizer]]
- randomly masking during data preparation

2. RoBERTa (Robustly optimized BERT approach):
- BPE  [[1708763669-GKCN|tokenizer]] --> higher number of subwords
- Dynamic randomly masking at training time instead of in data preparation step --> Diversity and Generalization

([PhoBert](https://github.com/VinAIResearch/PhoBERT) used RoBERTa architecture.)

3. DistilBERT (Distilled BERT): smaller, distilled version of BERT designed for efficiency.

4. ALBERT (A Lite BERT): focuses on model parameter reduction and achieves efficiency improvements by sharing parameters across layers. It maintains or surpasses BERT's performance with significantly fewer parameters.

6. XLNet: combines ideas from autoregressive language modeling (as seen in models like GPT) and BERT-style bidirectional context understanding. It achieves state-of-the-art performance by capturing both causal and bidirectional relationships.

7. ERNIE (Enhanced Representation through knowledge integration): incorporates knowledge graph information during pre-training to enhance representations. 

8. BioBERT: is specifically designed for biomedical text. It is pre-trained on biomedical literature and outperforms general-domain models when applied to biomedical NLP tasks.

## Reference
https://heidloff.net/article/foundation-models-transformers-bert-and-gpt/
https://www.comet.com/site/blog/roberta-a-modified-bert-model-for-nlp/
