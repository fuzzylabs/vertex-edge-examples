from edge.train import Trainer

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

class MyTrainer(Trainer):
    def main(self):
        self.set_parameter("epochs", 500)

        model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer="sgd", loss="mean_squared_error")

        model.fit(
            xs,
            ys,
            self.get_parameter("epochs"),
            callbacks = [
                LambdaCallback(on_epoch_end=self.log_epoch)
            ]
        )

        model.save(self.get_model_save_path())

        score = model.evaluate(xs, ys, verbose=0)
        return score

    def log_epoch(self, epoch, logs):
        self.log_scalar("loss", float(logs.get('loss')))

MyTrainer("hello-world").run()
