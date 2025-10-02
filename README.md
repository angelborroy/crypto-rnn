`crypto-rnn`: Learning the Enigma with Recurrent Neural Networks
=======
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

---

## Modifications from the Original Work

This branch, **`tf-2.20`**, is intended to upgrade the original environment to Python 3.11 and TensorFlow 2.20.

A new **Dockerfile** is included to run this environment.

### Build the Docker image

```bash
docker build --platform=linux/arm64 -t crypto-rnn-tf220 . --load
```

### Run the container (from the project root)

```bash
docker run --rm -it -v "$PWD:/app" crypto-rnn-tf220
```

* If using Windows PowerShell: `docker run --platform=linux/amd64 -it --rm -v ${PWD}:/app crypto-rnn-tf220`
* If using Windows CMD: `docker run --platform=linux/amd64 -it --rm -v %cd%:/app crypto-rnn-tf220`

### Train (example: Vigenère)

```bash
python main.py --cipher vigenere --total_steps 1000 --rnn_size 256 --tsteps 20 \
  --acc_every 500 --lr 5e-4 --batch_size 50 --key_len 6
```

Training artifacts are written to the `meta/` and `save/` directories.

### Summarize results

A helper script is provided to summarize a run:

```bash
python summarize_stats.py
```

Example output:

```json
{
  "total_steps": 1000,
  "final_loss": 231.5840244293213,
  "final_accuracy": 7.099999999999999,
  "total_training_time_seconds": 46.44298839569092,
  "avg_time_per_step": 0.046442988395690915
}
```
