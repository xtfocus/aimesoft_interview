---
id: 1708757466-HGNK
aliases:
  - teacher forcing
tags: []
---

# Teacher Forcing

In Teacher Forcing, instead of using the decoder's own **generated** output from the previous time step as input for the next time step, the **actual** (or teacher) values from the training dataset are used as input. The idea is to guide the decoder with the correct sequence during training, providing it with the ground truth information at each step.

Advantages of Teacher Forcing:
- Stability: It helps in stabilizing the training process by providing consistent and correct inputs during the training phase.
- Faster Convergence: Teacher Forcing can lead to faster convergence during training.

Considerations:
- While teacher forcing is beneficial during training, it may introduce a mismatch between training and inference, as the model is not exposed to its own mistakes during training.
- To address this mismatch, techniques such as scheduled sampling or a mix of teacher forcing and using the model's own output during training can be employed.

# Reference

https://blog.floydhub.com/attention-mechanism/

