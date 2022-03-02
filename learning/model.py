import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import Model, Sequential
from tensorflow import keras
from tensorflow.keras import layers


def conv_layer(inputs, filters, dropout):
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same")(inputs)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def baseline_model(input_shape, filters, dropout, mlp_units, mlp_dropout):
    

    input_layer = tf.keras.layers.Input(input_shape)

    conv1 = conv_layer(input_layer, filters, dropout)
    conv2 = conv_layer(conv1, filters, dropout)
    conv3 = conv_layer(conv2, filters, dropout)
    conv4 = conv_layer(conv3, filters, dropout)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(conv4)

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    

    output_layer = tf.keras.layers.Dense(2, activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    n_classes = 2
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)



def lstm_model(input_shape, n_lstm_nodes, n_lstm_layers, mlp_units, dropout, mlp_dropout):


    inputs = keras.Input(shape=input_shape)
    x = inputs

    for _ in range(n_lstm_layers):
        x = LSTM(units=n_lstm_nodes, dropout=dropout, return_sequences=True)(x)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    n_classes = 2
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

