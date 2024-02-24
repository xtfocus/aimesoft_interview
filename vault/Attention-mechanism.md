---
id: 1708746993-MQOD
aliases:
  - attention mechanism
  - attention
tags: []
---

# Attention mechanism

Attention mechanism was invented by Dzmitry Bahdanau's [paper](https://arxiv.org/pdf/1409.0473.pdf) on Machine Translation in 2015.

>*The standard seq2seq model is generally unable to accurately process long input sequences, since only the last hidden state of the encoder RNN is used as the context vector for the decoder. On the other hand, the Attention Mechanism directly addresses this issue as it retains and utilises all the hidden states of the input sequence during the decoding process. It does this by creating a unique mapping between each time step of the decoder output to all the encoder hidden states. This means that for each output that the decoder makes, it has access to the entire input sequence and can selectively pick out specific elements from that sequence to produce the output.* (Gabriel Loye, in a blogpost)

[Dzmitry Bahdanau](https://www.linkedin.com/in/dzmitry-bahdanau-a716b391/) seems to be a pretty young guy.

# Self-Attention (in Encoder)

In 2017, the groundbreaking paper "Attention is All You Need" introduced the transformer architecture, featuring a crucial mechanism known as self-attention in the Encoder. This self-attention mechanism is aptly named because it operates on the input sequence itself to learn embeddings. Each token within the sequence can play the roles of query, key, and value, contributing to a comprehensive understanding of contextual relationships.


# Multi-Head

An attention head consists of parameters Q, K, and V respectively represent query, key, and value. A set of parameters is called a head. You can use multiple sets to enhance representation, i.e., Multi-Head Self-Attention, Masked Multi-Head Self-Attention, even Multi-Head Cross-Attention

# Masked (in Decoder)

Masking is needed to make sure the model doesn't peek into subsequent timesteps during training the decoder. Hence, the name: Masked Multi-Head Self-Attention

# Cross-Attention (in Decoder)
aka Encoder-Decoder Attention

The output of the encoder is used as keys and values, and the decoder generates queries. This allows the decoder to attend to different parts of the input sequence (encoder's output) when generating each word in the target sequence.

# Reference
https://blog.floydhub.com/attention-mechanism/
