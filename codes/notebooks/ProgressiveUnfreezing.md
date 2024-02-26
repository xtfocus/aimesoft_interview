You have the option to free of unfree any parameters any time you want, by modifying `requires_grad`, like so:

```python
# Freeze the parameters of bert base
for param in model.bert.parameters():
    param.requires_grad = False

```

In practice, people usually freeze the base layers for a few initial epochs, then unfreeze them for the rest of the training. The motivation is to warm up the classification head, training to its max potential, then train the base layers.

# Reference

https://discuss.huggingface.co/t/gradual-unfreezing-support-for-fine-tuning-models/860/2

[reddit](https://www.reddit.com/r/MachineLearning/comments/ohr572/d_is_gradual_unfreezing_still_used/)

[wrt fast.ai](https://www.reddit.com/r/MachineLearning/comments/mbhewa/d_advanced_takeaways_from_fastai_book/)
PhoBert Sentiment Analysis example [train.py](https://github.com/suicao/PhoBert-Sentiment-Classification/blob/master/train.py)
