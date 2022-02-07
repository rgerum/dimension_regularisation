import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import time
from .robustness import robust_test


class PlotAlpha(keras.callbacks.Callback):
    def __init__(self, output, x_train, batch_size=1000, download_dir=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.output = output / "data.csv"
        self.output2 = output / "alpha.csv"
        self.model_save = output / "model_save"
        self.state_output = output / "status.txt"
        self.x_train = x_train
        self.batch_size = batch_size

        self.download_dir = download_dir

        self.data = []
        self.alpha_data = []

    def started(self):
        if Path(self.output).exists():
            try:
                history = pd.read_csv(self.output)
                history.epoch.max()
                model = tf.keras.models.load_model("tmp_history")
                initial_epoch = int(history.epoch.max() + 1)
                data = [dict(history.iloc[i]) for i in range(len(history))]
                self.start_data = dict(model=model, initial_epoch=initial_epoch, data=data)
                return True
            except:
                return False
        return False

    def load(self):
        return self.start_data["model"], self.start_data["initial_epoch"]

    def on_epoch_end(self, epoch, logs={}):
        try:
            from slurm_job_submitter import set_job_status
            set_job_status(dict(epoch=epoch))
        except ModuleNotFoundError:
            pass

        for mode in ["brightness", "contrast", "defocus_blur", "elastic", "gaussian_noise"]:
            for i in range(1, 6):
                logs[f"accuracy_{mode}_{i}"] = robust_test(self.model, mode, i, self.download_dir)
        logs["epoch"] = epoch
        logs["time"] = time.time()
        self.data.append(logs)

        Path(self.model_save).mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_save)

        eigen_values_list = []
        names = []
        for i in range(100):
            j = 0
            model2 = keras.models.Sequential()
            for layer in self.model.layers:
                if isinstance(layer, DimensionReg):
                    if j == i:
                        model2.add(DimensionRegOutput())
                        name = layer.metric_name
                        names.append(layer.metric_name)
                        break
                    else:
                        j += 1
                if isinstance(layer, keras.layers.Dropout):
                    rate = layer.rate
                    model2.add(keras.layers.Lambda(lambda x: tf.nn.dropout(x, rate=rate)))
                else:
                    model2.add(layer)
            else:
                break
            eigen_values = model2(self.x_train[:self.batch_size]).numpy()
            eigen_values_list.append(eigen_values)
            self.alpha_data.extend([dict(epoch=epoch, name=name, value=x) for x in eigen_values])
        while True:
            try:
                pd.DataFrame(self.data).to_csv(self.output, index=False)
            except FileNotFoundError as err:
                print(err)
                self.output.parent.mkdir(parents=True, exist_ok=True)
            else:
                break
        while True:
            try:
                pd.DataFrame(self.alpha_data).to_csv(self.output2, index=False)
            except FileNotFoundError as err:
                print(err)
                self.output2.parent.mkdir(parents=True, exist_ok=True)
            else:
                break
        with Path(self.state_output).open("w") as fp:
            fp.write(f"{self.data[-1]['epoch']}\n")


    def on_train_end(self, logs=None):
        with Path(self.state_output).open("w") as fp:
            fp.write(f"{self.data[-1]['epoch']} done\n")
