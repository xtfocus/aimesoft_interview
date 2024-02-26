---
id: 1708928565-THSB
aliases:
  - Gradient Accumulation
  - 1708928538-PYZB
tags: []
---

# Gradient Accumulation
## Motivation
In reality, the full batch gradient is often not possible to compute since holding the full dataset in memory is infeasible

## Theory
Split a batch (to be called the full batch, or the global batch) into multiple *mini-batches*. Then, rather than updating the whole batch, accumulate all gradients from mini-batches before updating.

Mini-batches provide an approximation of the true gradient.

## Practical
If you are rich on VRAM, reduce the number of accumulation steps to make the weight update more frequent.
There are still some discussion on gradient accumulation (see Reference:Reddit).

# Reference
[Simple explanation](https://ai.stackexchange.com/questions/38895/why-to-use-gradient-accumulation)
[In depth explanation](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)
[Manual implementation](https://towardsdatascience.com/how-to-easily-use-gradient-accumulation-in-keras-models-fa02c0342b60)
[In-depth discussion](https://www.reddit.com/r/MachineLearning/comments/wxvlcc/d_does_gradient_accumulation_achieve_anything/)
