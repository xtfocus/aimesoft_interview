# (possible) BERT related interview topics 

Other than the big picture of transformers, some details you are expected to know as a NLP specialist:

1. About `[CLS]` token

Also called the classification token. Why is it at the start of the embedding output: It actually doesn't mattter start or end because it's not a recurrent model. 

You also have to option to not only use the last layer but also earlier layers.

A notebook about CLS token as BERT's output is under `codes/notebooks/BertModel outputs and CLS token.ipynb` of this repo!


2. About tokenization
How does tokenization works in BERT? Name other tokenization techniques. What's the tokenization in PhoBert?

3. About accumulation grad updates
What is it for? How does it work?

4. How's *layer normalization* in BERT works? How is it compared to batch normalization

5. Explain the use of Adam optimizer and scheduler's parameters in detail. What's the number of training steps? What's an epoch

Reference
[stackexchange CLS token](https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-is-its-encoding-output-important)
[huggingface's issue on CLS token](https://github.com/huggingface/transformers/issues/7540)

