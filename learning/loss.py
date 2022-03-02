import tensorflow as tf


def crossentropy_loss(model_output, target, irrig_lm, noirrig_lm):
    

    weights = irrig_lm * target + noirrig_lm * (1-target)

    loss_fx = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                label_smoothing = 0.2,
                                                from_logits=False)

    loss = loss_fx(target, model_output, sample_weight=weights)

    return loss
