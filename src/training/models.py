from typing import Dict
import tensorflow as tf
import tensorflow_io
from tensorflow.keras import layers, Model, Input
from constants import *
from prepare_dataset import position_encoding
from scalers import robust_scaler, max_scaling

class CustomScaling(layers.Layer):
    def __init__(self, name):
        super().__init__()
        if name == 'max':
            self.scaler = max_scaling
        elif name == 'robust':
            self.scaler = robust_scaler


    def call(self, history_channels, epsilon):
        return self.scaler(history_channels, epsilon)

class PositionExpansion(layers.Layer):
    def __init__(self, periods: int, freqs: int, **kwargs):
        super().__init__(**kwargs)
        # Channels could be ceiling(log_2(periods))
        self.periods = periods
        self.channels = freqs * 2
        self.embedding = tf.constant(position_encoding(periods, freqs))

    def call(self, tc):
        flat = tf.reshape(tc, [1, -1])
        embedded = tf.gather(self.embedding, flat)
        out_shape = tf.shape(tc)
        return tf.reshape(embedded, [out_shape[0], out_shape[1], self.channels])

class TransformerBlock(layers.Layer):
    def __init__(self, key_dim, heads=4, value_dim=None, residual=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=heads, key_dim=key_dim, value_dim=value_dim, name=f'{self.name}_attention')
        value_dim = value_dim or key_dim
        self.ff1 = layers.Dense(4 * heads * value_dim, activation='gelu', name=f'{self.name}_ff1')
        self.ff2 = layers.Dense(heads * value_dim, activation='gelu', name=f'{self.name}_ff2')
        self.residual = residual
        if self.residual:
            self.attn_norm = layers.LayerNormalization(name=f'{self.name}_attn_norm')
            self.ff_norm = layers.LayerNormalization(name=f'{self.name}_ff_norm')

    def build(self, input_shapes):
        self.attention._build_from_signature(input_shapes, input_shapes)

    def call(self, x, mask):
        a = self.attention(x, x, attention_mask=mask)
        a = self.ff1(a)
        return self.ff2(a)
        #na = self.attn_norm(a + x)
        #return self.ff_norm(self.ff(na) + na)

class BaseModel(tf.keras.Model):
    def __init__(self, epsilon=1e-4, scaler='robust', **kwargs):
        super().__init__()
        self.epsilon = epsilon
        self.pos_year = PositionExpansion(10, 4)
        self.pos_month = PositionExpansion(12, 4)
        self.pos_day = PositionExpansion(31, 6)
        self.pos_dow = PositionExpansion(7, 4)
        self.robust_scaler = CustomScaling(scaler)
        self.embed_size = sum(emb.channels for emb in (self.pos_year, self.pos_month, self.pos_day, self.pos_dow))
        self.expand_target_nopos = layers.Dense(self.embed_size, name='NoPosEnc', activation='relu')
        self.expand_target_forpos = layers.Dense(self.embed_size, name='ForPosEnc', activation='relu')
        self.concat_pos = layers.Concatenate(axis=-1, name='ConcatPos')
        self.concat_embed = layers.Concatenate(axis=-1, name='ConcatEmbed')
        # Will be an embedding when we have different tasks.
        self.target_marker = layers.Embedding(NUM_TASKS, self.embed_size)

    @staticmethod
    def tc(ts: tf.Tensor, time_index: int):
        return ts[:, :, time_index]

    def call(self, x: Dict[str, tf.Tensor]):
        ts, history, target_ts, task = x['ts'], x['history'], x['target_ts'], x['task']
        print(ts.shape, ts)

        # Build position encodings
        year = self.tc(ts, YEAR)
        delta_year = tf.clip_by_value(year[:, -1:] - year, 0, self.pos_year.periods)
        pos_embedding = self.concat_pos([
            self.pos_year(delta_year),
            self.pos_month(self.tc(ts, MONTH)),
            self.pos_day(self.tc(ts, DAY)),
            self.pos_dow(self.tc(ts, DOW)),
        ])
        mask = year > 0

        # Embed history
        history_channels = tf.expand_dims(history, axis=-1)
#         scale = self.max_scaling(history_channels) + self.epsilon
#         scaled = history_channels / scale
        scale, scaled = self.robust_scaler(history_channels, self.epsilon)
        embed_nopos = self.expand_target_nopos(scaled)
        embed_pos = self.expand_target_forpos(scaled) + pos_embedding
        embedded = self.concat_embed([embed_nopos, embed_pos])


        # Embed target
        target_year = tf.clip_by_value(year[:, -1:] - self.tc(target_ts, YEAR), 0, self.pos_year.periods)
        target_pos_embed = tf.squeeze(self.concat_pos([
            self.pos_year(target_year),
            self.pos_month(self.tc(target_ts, MONTH)),
            self.pos_day(self.tc(target_ts, DAY)),
            self.pos_dow(self.tc(target_ts, DOW)),
        ]), axis=1)
        task_embed = self.target_marker(task)
        target = self.concat_embed([task_embed, task_embed + target_pos_embed])

        result = self.forecast(ts, mask, scale, embedded, target)
        scale = tf.squeeze(scale, axis=-1)
        result = result * scale
        return {'result': result, 'scale': scale}

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # return super().compute_loss(x, y, y_pred['result'], sample_weight)
        scale = y_pred['scale']
        return super().compute_loss(x, y / scale, y_pred['result'] / scale, sample_weight)

    def forecast(self, ts: tf.Tensor, mask: tf.Tensor, scale: tf.Tensor, embedded: tf.Tensor, target: tf.Tensor):
        return NotImplemented

class LSTMModel(BaseModel):
    def __init__(self, unit=30, **kwargs):
        super().__init__(**kwargs)
        self.units = 30
        self.lstm = layers.LSTM(self.units)
        self.combine_target = layers.Concatenate(name='AppendTarget', axis=-1)
        self.cont_output = layers.Dense(1, name='Output', activation='relu')

    def forecast(self, ts: tf.Tensor, mask: tf.Tensor, scale: tf.Tensor, embedded: tf.Tensor, target: tf.Tensor):
        lstm_out = self.lstm(embedded, mask=mask)
        with_target = self.combine_target([lstm_out, target])
        return self.cont_output(with_target)


class TransformerModel(BaseModel):
    def __init__(self, tx_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.tx_layers = tx_layers
        self.concat_target = layers.Concatenate(name='AppendTarget', axis=1)
        self.encoder1 = TransformerBlock(key_dim=(self.embed_size * 2))
        self.encoder2 = TransformerBlock(key_dim=(self.embed_size * 2))
        # self.encoder3 = TransformerBlock(key_dim=(self.embed_size * 2))
        self.final_output = layers.Dense(1, name='FinalOutput', activation='relu')

    # def __init__(self, tx_layers=2, **kwargs):
    #     super().__init__(**kwargs)
    #     self.tx_layers = tx_layers
    #     self.concat_target = layers.Concatenate(name='AppendTarget', axis=1)
    #     heads = 4
    #     k_d = v_d = (2 * self.embed_size) // heads
    #     self.encoder1 = TransformerBlock(key_dim=k_d, heads=heads, value_dim=v_d)
    #     # self.encoder1 = TransformerBlock(key_dim=(self.embed_size * 2))
    #     self.encoder2 = TransformerBlock(key_dim=(self.embed_size * 2))
    #     self.final_output = layers.Dense(1, name='FinalOutput', activation='relu')

    def forecast(self, ts: tf.Tensor, mask: tf.Tensor, scale: tf.Tensor, embedded: tf.Tensor, target: tf.Tensor):
        mask = tf.pad(mask, [[0, 0], [0, 1]], constant_values=True)
        mask = tf.math.logical_and(tf.expand_dims(mask, 1), tf.expand_dims(mask, -1))
        x = self.concat_target([
            embedded,
            tf.expand_dims(target, axis=1)
        ])
        x = self.encoder1(x, mask)
        x = self.encoder2(x, mask)
        # x = self.encoder3(x, mask)
        x = self.final_output(x[:, -1, :])
        return x
