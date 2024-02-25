
# NLP Interview Refresher Repository

I have created this repository for revision purposes, especially for an upcoming NLP interview. As part of a skill test for a company, I will be performing Bert Sentiment Classification on the [restaurant review dataset](https://www.kaggle.com/datasets/vigneshwarsofficial/reviews/data)

While there are numerous tutorials available on Kaggle for this challenge dating back to 2020, many of them utilize Transformer v3 (the latest version is v4). Some tutorials include explanations about the Transformer concept, but a significant number do not.

The primary goal of this repository is to fill in knowledge gaps and provide learners with a comprehensive perspective on BERT, Transformers, and the Hugging Face library, which are frequently discussed concepts in NLP interviews.

The study notes reside in the `vault` directory, meant to be opened with [obsidian](https://obsidian.md/download). This setup minimizes friction when introducing new concepts. You can seamlessly explore side notes, read, close them, and return to the main note. During subsequent reviews, familiarity with the side notes allows for efficient learning—skipping already grasped content. This approach offers multiple benefits without disrupting traditional learning methods. The content of `vault` is high-level, which I found suitable for interview revision. However, where possible, I leave pointers in case I want to deep dive into a concept later.

The ML codes are in the `code` directory. For learners, it is recommended to open then in [Colab](https://colab.research.google.com/) for minimum friction.

Feel free to explore the repository and leverage its contents for your own NLP interview preparation!

# TODO

- [x] codes: demo tokenizers and CLS based on https://github.com/huggingface/transformers/issues/7540 and https://discuss.huggingface.co/t/how-to-get-cls-embeddings-from-bertfortokenclassification-model/9276/2

in `codes/notebooks/BertModel outputs and CLS token.ipynb`


- [ ] vault: Training details: Optimizer and Scheduler

- [ ] experiment agenda and implementaion: MlOps
