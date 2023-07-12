import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
import random

# fix seeds
seed = 5262668
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

MAX_TOKENS = 128
BUFFER_SIZE = 2000
BATCH_SIZE = 64

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth

    angle_rates = 1/(10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, is_mp: bool):
        super().__init__()
        self.d_model = d_model
        if not is_mp:
            self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        else:
            self.linear_mapping = keras.layers.Dense(d_model, activation='linear')
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)
        self.is_mp = is_mp

    def compute_mask(self, *args, **kwargs):
        if self.is_mp:
            return super().compute_mask(*args, **kwargs)
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, *args, **kwargs):
        length = tf.shape(x)[1]
        if not self.is_mp:
            x = self.embedding(x)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        else:
            x = self.linear_mapping(x)
        #print(x.shape)
        #print(self.pos_encoding[tf.newaxis, :length, :].shape)
        x = x + self.pos_encoding[tf.newaxis, :length, :x.shape[-1]]
        return x

class BaseAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):

    def call(self, x, context, *args, **kwargs):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

class GlobalSelfAttention(BaseAttention):

    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class CausalSelfAttention(BaseAttention):

    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):

    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, *args, **kwargs):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model
        )
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, *args, **kwargs):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):

    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, droput_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model, is_mp=True
        )

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=droput_rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(droput_rate)

    def call(self, x, *args, **kwargs):
        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model
        )

        self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def call(self, x, context, *args, **kwargs):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x

class Decoder(tf.keras.layers.Layer):

    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model, is_mp=False)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context, *args, **kwargs):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x


class Transformer(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, vocab_size=None, droput_rate=dropout_rate, dff=dff)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=target_vocab_size, dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)  # Why no Softmax??

    def call(self, inputs, *args, **kwargs):
        mp_data, previous_prediction = inputs

        x_enc = self.encoder(mp_data)

        x = self.decoder(previous_prediction, x_enc)

        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


def get_model(input1_shape, input2_shape, num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate=0.1):
    in1 = tf.keras.layers.Input(input1_shape)
    in2 = tf.keras.layers.Input(input2_shape)

    transformer = Transformer(num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate)

    out = transformer([in1, in2])

    return keras.Model([in1, in2], out)


class CustomTransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,

        }
        return config


def masked_loss(label, pred):
    mask = label != 59
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_obj(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, dtype=pred.dtype)
    match = label == pred

    mask = label != 59
    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


def get_compiled_transformer(sign_shape: tuple, context_shape: tuple, d_model: int, num_layers: int, num_heads: int, ff_dim: int, output_vocab_size: int, dropout_rate: float = 0.1) -> Transformer:
    learning_rate = CustomTransformerSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer = get_model(
        input1_shape=sign_shape,
        input2_shape=context_shape,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=ff_dim,
        target_vocab_size=output_vocab_size,
        dropout_rate=dropout_rate
    )
    transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
    return transformer
