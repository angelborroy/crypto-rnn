# Learning the Enigma with Recurrent Neural Networks
# Sam Greydanus. January 2017. MIT License.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras

class StackedRNN:
    """
    TF 2.20 / Keras 3 compatible stacked LSTM that preserves:
      - TF1 placeholders (x, y_) and graph mode
      - logits/y tensors, loss, train_op
      - Saver-based save/load
      - API used by train_utils.py: count_params, try_load_model, train_step, eval_step, predict, decode, reset_states, step
    """

    def __init__(self, FLAGS, crack_mode=False):
        # Keep TF1 graph/session style so the rest of the repo runs unchanged
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.InteractiveSession()

        self.FLAGS = FLAGS
        self.scope = "rnn"
        self.crack_mode = crack_mode

        # Core flags used across the repo
        self.batch_size = FLAGS.batch_size
        self.rnn_size   = FLAGS.rnn_size
        self.tsteps     = FLAGS.tsteps
        self.xlen       = 27  # alphabet size used in original code
        self.ylen       = 27
        self.num_layers = getattr(FLAGS, "num_layers", 1)

        # ---------------------------
        # Placeholders (TF1 style)
        # x: [B, T, xlen], y_: [B, T, ylen]
        self.x  = tf.compat.v1.placeholder(tf.float32, shape=[None, self.tsteps, self.xlen], name="x")
        self.y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, self.tsteps, self.ylen], name="y")

        # ---------------------------
        # Build stacked LSTMs with Keras layers (Keras 3 compatible)
        init = keras.initializers.TruncatedNormal(stddev=0.1)  # dtype passed at call sites
        self.rnn_layers = [
            keras.layers.LSTM(
                units=self.rnn_size,
                return_sequences=True,
                return_state=True,
                kernel_initializer=init,
                name=f"lstm_{i}",
            )
            for i in range(self.num_layers)
        ]

        def forward(x):
            """
            Stateless per-batch forward: zero init each batch,
            matching TF1 dynamic_rnn default (no cross-batch state carry).
            """
            seq = x
            batch = tf.shape(seq)[0]
            for layer in self.rnn_layers:
                h0 = tf.zeros([batch, self.rnn_size], dtype=seq.dtype)
                c0 = tf.zeros([batch, self.rnn_size], dtype=seq.dtype)
                seq, _, _ = layer(seq, initial_state=[h0, c0])
            return seq  # [B, T, H]

        outputs = forward(self.x)  # [B, T, H]

        # ---------------------------
        # Readout (time-distributed Dense as flatten + matmul)
        with tf.compat.v1.variable_scope(self.scope, reuse=False):
            W = tf.compat.v1.get_variable(
                "W_out",
                shape=[self.rnn_size, self.ylen],
                initializer=tf.compat.v1.truncated_normal_initializer(
                    stddev=1.0 / np.sqrt(self.rnn_size), dtype=tf.float32
                ),
            )
            b = tf.compat.v1.get_variable(
                "b_out",
                shape=[self.ylen],
                initializer=tf.compat.v1.constant_initializer(0.0),
            )

        BT = tf.shape(outputs)[0] * tf.shape(outputs)[1]
        H  = tf.shape(outputs)[2]
        flat = tf.reshape(outputs, [BT, H])
        logits_flat = tf.matmul(flat, W) + b
        self.logits = tf.reshape(logits_flat, [-1, self.tsteps, self.ylen], name="logits")
        self.y = tf.nn.softmax(self.logits, name="y_softmax")

        # ---------------------------
        # Loss + Optimizer (TF1 style)
        loss_t = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits)
        self.loss = tf.reduce_mean(loss_t, name="loss")
        opt = tf.compat.v1.train.AdamOptimizer(self.FLAGS.lr)
        self.train_op = opt.minimize(self.loss)

        # ---------------------------
        # STEP GRAPH for sampling (stateful across calls, batch can be 1)
        # Reuse the same LSTM weights and output weights; run on a single timestep.
        self.x_step = tf.compat.v1.placeholder(tf.float32, shape=[None, 1, self.xlen], name="x_step")

        # One (h,c) input placeholder per layer
        self.h_in, self.c_in = [], []
        seq = self.x_step
        for i, layer in enumerate(self.rnn_layers):
            self.h_in.append(tf.compat.v1.placeholder(tf.float32, shape=[None, self.rnn_size], name=f"h_in_{i}"))
            self.c_in.append(tf.compat.v1.placeholder(tf.float32, shape=[None, self.rnn_size], name=f"c_in_{i}"))
            seq, h, c = layer(seq, initial_state=[self.h_in[i], self.c_in[i]])
            # collect outputs for state update
            if i == 0:
                self.h_out, self.c_out = [h], [c]
            else:
                self.h_out.append(h)
                self.c_out.append(c)

        # Readout for the last timestep of the step graph: [B, H] -> [B, ylen]
        last_h = seq[:, -1, :]                       # [B, H]
        logits_step = tf.matmul(last_h, W) + b       # reuse W, b
        self.y_step = tf.nn.softmax(logits_step, name="y_step")

        # Python-side cache of current recurrent state for sampling
        self._step_state = None

        # ---------------------------
        # Saver and session init
        self.save_path = os.path.join(self.FLAGS.save_dir, "model.ckpt")
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        self.sess.run(tf.compat.v1.global_variables_initializer())

    # ---------------------------
    # Utilities expected by the rest of your code

    def count_params(self):
        """Print and return the total number of trainable parameters."""
        total = 0
        for v in tf.compat.v1.trainable_variables():
            shape = v.shape.as_list()
            n = 1
            for d in shape:
                n *= int(d)
            total += n
        try:
            print(f"Model parameters: {total:,}")
        except Exception:
            print("Model parameters:", total)
        return int(total)

    def try_load_model(self):
        """Try to restore latest checkpoint and return global_step (int)."""
        global_step = 0
        try:
            save_dir = os.path.dirname(self.save_path)
            ckpt = tf.compat.v1.train.get_checkpoint_state(save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                load_path = ckpt.model_checkpoint_path
                self.saver.restore(self.sess, load_path)
                print(f"loaded model: {load_path}")
                # refresh saver to bind to current graph variables
                self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
                try:
                    global_step = int(load_path.split('-')[-1])
                except Exception:
                    global_step = 0
            else:
                print("no saved model to load. starting new session")
        except Exception:
            print("no saved model to load. starting new session")
        return global_step

    # ---------------------------
    # Steps expected by train_utils.py

    def _parse_batch(self, batch):
        """
        Accept batch as (x, y), [x, y], or dict with keys ('x','y') or ('x','y_').
        Shapes expected: x=[B,T,xlen], y=[B,T,ylen].
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            return x, y
        if isinstance(batch, dict):
            if 'x' in batch and 'y_' in batch:
                return batch['x'], batch['y_']
            if 'x' in batch and 'y' in batch:
                return batch['x'], batch['y']
        raise ValueError("Unsupported batch format for StackedRNN")

    def train_step(self, batch):
        """Run one training step and return the scalar loss."""
        x, y = self._parse_batch(batch)
        feed = { self.x: x, self.y_: y }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        return float(loss)

    def eval_step(self, batch):
        """Compute loss (no weight updates)."""
        x, y = self._parse_batch(batch)
        feed = { self.x: x, self.y_: y }
        loss = self.sess.run(self.loss, feed_dict=feed)
        return float(loss)

    def predict(self, batch):
        """Return softmax outputs y for a batch."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
        elif isinstance(batch, dict) and 'x' in batch:
            x = batch['x']
        else:
            raise ValueError("Unsupported batch format for predict()")
        feed = { self.x: x }
        y = self.sess.run(self.y, feed_dict=feed)
        return y

    def decode(self, x):
        """
        Given input x [B,T,xlen], return one-hot predictions [B,T,ylen].
        Matches original accuracy() expectation.
        """
        y_prob = self.predict({'x': x})
        idx = np.argmax(y_prob, axis=2)
        one = np.zeros_like(y_prob)
        for b in range(y_prob.shape[0]):
            for t in range(y_prob.shape[1]):
                one[b, t, idx[b, t]] = 1.0
        return one

    # ---------------------------
    # Stateful sampling helpers used by sample()

    def reset_states(self, batch_size: int = 1):
        """Reset internal sampling state (default B=1)."""
        zeros = np.zeros((batch_size, self.rnn_size), dtype=np.float32)
        self._step_state = [(zeros.copy(), zeros.copy()) for _ in range(self.num_layers)]

    def step(self, x, return_state: bool = False):
        """
        Advance sampling by one timestep.
        x shape must be [B, 1, xlen]. Returns y_last [B, ylen] and (optionally) new state.
        """
        if self._step_state is None:
            # initialize if user forgot to call reset_states()
            self.reset_states(batch_size=x.shape[0])

        feed = { self.x_step: x }
        # feed current (h,c) into the placeholders
        for i, (h, c) in enumerate(self._step_state):
            feed[self.h_in[i]] = h
            feed[self.c_in[i]] = c

        fetch = [self.y_step] + self.h_out + self.c_out
        out = self.sess.run(fetch, feed_dict=feed)

        y_last = out[0]
        y_last_3d = y_last[:, None, :]
        hs = out[1:1+self.num_layers]
        cs = out[1+self.num_layers:1+2*self.num_layers]
        self._step_state = list(zip(hs, cs))

        if return_state:
            return y_last_3d, self._step_state
        return y_last_3d

    # ---------------------------
    # Saver helpers

    def load(self):
        """Load latest checkpoint if present; return True on success."""
        try:
            save_dir = os.path.dirname(self.save_path)
            ckpt = tf.compat.v1.train.get_checkpoint_state(save_dir)
            if not ckpt or not ckpt.model_checkpoint_path:
                print("no saved model to load. starting new session")
                return False
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
            print(f"loaded model: {load_path}")
            return True
        except Exception:
            print("no saved model to load. starting new session")
            return False

    def save(self, global_step):
        self.saver.save(self.sess, self.save_path, global_step=global_step)