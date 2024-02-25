# Notebooks to demo BERT concepts

# Experimental tracking

Agenda:

- Same optimizer and scheduler for runs. ([What's a MlFlow run ?](https://www.run.ai/guides/machine-learning-operations/mlflow#:~:text=A%20run%20is%20a%20collection,basic%20unit%20of%20MLflow%20organization.)):
    In reality, optimizers and schedulers also has parameters that can be tweaked for better performance.\
    For simplicity, we keep it simply the same among runs.

- Using the same BERT base (`bert-base-cased`), compare the performance of different classifier head architectures:

    There are multiple ideas, but for the demonstration purpose, I will be implementing the following classifier heads:

    - simple nn.Layer with [CLS] of the last layer's hidden state as input
    - simple nn.Layer with the untrained [CLS] of the last layer's hidden state as input (which is the first element of `last_hidden_state`)
    - simple nn.Layer with of the concatinated of last `n` layers' first element
