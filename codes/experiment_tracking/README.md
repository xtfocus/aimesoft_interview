# Experiment tracking dir

Here, I implement different choices of BERT-based sentiment classification, save metrics and parameters to a database using [MlFlow](https://mlflow.org/docs/latest/python_api/mlflow.html)

This directory is to demontrate the experiment-tracking concept in MLOps. In production, it is not the best practice to implement MlFlow with notebooks and sqlite dumps like I do here.

Also, since the emphasize is on MlFlow as an important tool for MLOps, the ML codes will be factorized for DRY sake. Which means the ML codes are not **necessary easy to read**. Readers still getting used to `transformers` and BERT concept can refer to notebooks under `codes/notebooks`.

