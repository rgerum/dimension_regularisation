import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import time
from .dimension_reg_layer import DimensionReg
from .dim_includes import DimensionRegOutput


class CalculateAlphaCurves(keras.callbacks.Callback):
    def __init__(self, output, x_train, batch_size=1000):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.output2 = output / "alpha.csv"
        self.x_train = x_train
        self.batch_size = batch_size

        self.alpha_data = []

    def on_epoch_end(self, epoch, logs={}):

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

        self.output2.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.alpha_data).to_csv(self.output2, index=False)

    def on_train_end(self, logs=None):
        with Path(self.state_output).open("w") as fp:
            fp.write(f"{self.data[-1]['epoch']} done\n")


class SaveHistory(keras.callbacks.Callback):
    def __init__(self, output, additional_logs_callback=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.filename_logs = output / "data.csv"
        self.filename_model = output / "model_save"

        self.logs_history = []
        self.additional_logs_callback = additional_logs_callback

    def started(self):
        # if the output file exists
        if Path(self.filename_logs).exists():
            # try to load it
            try:
                # read the saved history
                history = pd.read_csv(self.filename_logs)
                # get the last epoch
                initial_epoch = int(history.epoch.max() + 1)
                # convert the history back into a list of dictionaries
                data = [dict(history.iloc[i]) for i in range(len(history))]

                # load the model
                model = tf.keras.models.load_model("tmp_history")
                # store the new start data
                self.start_data = dict(model=model, initial_epoch=initial_epoch, data=data)
                return True
            # if anything goes wrong do not reload the model
            except Exception:
                return False
        # if the path does not exits we have no model to continue
        return False

    def load(self):
        self.logs_history = self.start_data["data"]
        return self.start_data["model"], self.start_data["initial_epoch"]

    def on_epoch_end(self, epoch, logs={}):
        # add epoch and time to the logs
        logs["epoch"] = epoch
        logs["time"] = time.time()
        # maybe add additional metrics
        if self.additional_logs_callback is not None:
            logs.update(self.additional_logs_callback(self.model))
        # store the logs
        self.logs_history.append(logs)

        # save the logs to a csv file
        self.filename_logs.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.logs_history).to_csv(self.filename_logs, index=False)

        # save the current model
        Path(self.filename_model).mkdir(parents=True, exist_ok=True)
        self.model.save(self.filename_model)


class SlurmJobSubmitterStatus(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        try:
            from slurm_job_submitter import set_job_status
            set_job_status(dict(epoch=epoch))
        except ModuleNotFoundError:
            pass