import numpy as np
import tensorflow as tf


def accuracy(y_true, y_pred):
    if getattr(y_true, "detach", None):
        y_true = y_true.detach()
    if getattr(y_pred, "detach", None):
        y_pred = y_pred.detach()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


def pgd(model, x, y, eps, noRestarts=1, lr=0.01, gradSteps=40):
    if not isinstance(eps, (list, tuple, np.ndarray)):
        eps = [eps]

    x_nats = []
    accs = []
    for e in eps:
        losses = [0]*noRestarts
        xs = []
        for r in range(noRestarts):
            if r == 0:
                perturb = 0
            else:
                perturb = 2 * e * tf.random.uniform(x.shape) - e
            x_start = x + perturb
            x_start = tf.clip_by_value(x_start, 0, 1)

            xT, ellT = pgd_tf(model, x, x_start, y, e, lr, gradSteps)  # do pgd
            xs.append(xT)
            losses[r] = ellT
        idx = np.argmax(losses)
        x = xs[idx]  # choose the one with the largest loss function
        ell = losses[idx]
        a = accuracy(y, model(x))
        accs.append(a)
    return accs
    return x, ell, a


def clip_tf(x, lb, ub):
    x = tf.where(x > ub, ub, x)
    x = tf.where(x < lb, lb, x)
    return x

def pgd_tf(model, x_nat, x, y, eps, lr, gradSteps):
    for i in range(gradSteps):
        # get jacobian
        im = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(im)
            prediction = model(im)
            # loss = tf.reduce_sum(- tf.reduce_sum(prediction * input_label, axis=1) + tf.math.log(tf.reduce_sum(tf.exp(prediction), axis=1)))
            prediction2 = tf.nn.softmax(prediction)
            loss = tf.keras.losses.CategoricalCrossentropy()(y, prediction2)
        # Get the gradients of the loss w.r.t to the input image.
        jacobian = tape.gradient(loss, im)

        xT = (x + lr * tf.sign(jacobian))
        xT = clip_tf(xT, x_nat - eps, x_nat + eps)

        # if just one channel, then lb and ub are just numbers
        xT = tf.clip_by_value(xT, 0, 1)

        x = xT

    prediction = model(x)
    # loss = tf.reduce_sum(- tf.reduce_sum(prediction * input_label, axis=1) + tf.math.log(tf.reduce_sum(tf.exp(prediction), axis=1)))
    prediction2 = tf.nn.softmax(prediction)
    loss = tf.keras.losses.CategoricalCrossentropy()(y, prediction2)
    return x, loss


def fgsm(model, x, y, eps):
    im = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(im)
        prediction = model(im)
        #loss = tf.reduce_sum(- tf.reduce_sum(prediction * input_label, axis=1) + tf.math.log(tf.reduce_sum(tf.exp(prediction), axis=1)))
        prediction2 = tf.nn.softmax(prediction)
        loss = tf.keras.losses.CategoricalCrossentropy()(y, prediction2)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, im)

    if not isinstance(eps, (list, tuple, np.ndarray)):
        eps = [eps]
    x_advs = []
    acc = []
    for e in eps:
        x_adv = x + e * tf.sign(gradient)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
        a = accuracy(y, model(x_adv))

        x_advs.append(x_adv)
        acc.append(a)
    return acc
    return gradient, loss, prediction, x_advs, acc


def get_tf_model(d):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=2000, activation='tanh'),
        tf.keras.layers.Dense(units=10),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.weights[0].assign(d["_batch_modifier._architecture.sequential.0.weight"].T)
    model.weights[1].assign(d["_batch_modifier._architecture.sequential.0.bias"])
    model.weights[2].assign(d["_batch_modifier._architecture.sequential.2.weight"].T)
    model.weights[3].assign(d["_batch_modifier._architecture.sequential.2.bias"])
    return model
