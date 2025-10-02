`crypto-rnn`: Learning the Enigma with Recurrent Neural Networks
=======

**Choose your TensorFlow version**

* **TF 1.1 (original code, Python 2.7/TF1 graph)**: use branch **[`tf-1.1`](https://github.com/angelborroy/crypto-rnn/tree/tf-1.1)**  
* **TF 2.20 (modernized, TF2 runtime in graph mode + Keras LSTM)**: use branch **[`tf-2.20`](https://github.com/angelborroy/crypto-rnn/tree/tf-2.20)**

The `master` branch contains general docs; for running code, **switch to the branch that matches your TF version**.

--

See [paper](https://arxiv.org/abs/1708.07576) and [blog post](https://greydanus.github.io/2017/01/07/enigma-rnn/)

![concept-small](static/concept-small.png)

About
--------
This repo contains a deep LSTM-based model for learning polyalphabetic ciphers. It also contains code for training the model on three ciphers: the Vigenere, Autokey, and Enigma ciphers. The first two are light proof-of-concept tasks whereas the Enigma is much more complex. For this reason, the Enigma model is enormous (3000 hidden units) and takes a lot longer to train.

Vigenere and Autokey ciphers
--------
The [Vigenere cipher](https://en.wikipedia.org/wiki/Vigen%C3%A8re_cipher) works like this (where we're encrypting plaintext "CALCUL" with keyword "MATHS" (repeated)). The [Autokey cipher](https://en.wikipedia.org/wiki/Autokey_cipher) is a slightly more secure variant.
![Vigenere cipher](static/vigenere.gif?raw=true)

Enigma cipher
--------
The [Enigma cipher](https://en.wikipedia.org/wiki/Enigma_machine) works like this.
![Enigma cipher](static/enigma.gif?raw=true)

Dependencies
--------
* All code is written in Python 3.6 and TensorFlow 1.1. You will need:
 * NumPy
 * [TensorFlow](https://www.tensorflow.org/install/)

### Branches

- **TF 1.1 (original):** [`tf-1.1`](https://github.com/angelborroy/crypto-rnn/tree/tf-1.1)  
- **TF 2.20 (updated):** [`tf-2.20`](https://github.com/angelborroy/crypto-rnn/tree/tf-2.20)

> Use the branch README for setup and run instructions specific to that version.