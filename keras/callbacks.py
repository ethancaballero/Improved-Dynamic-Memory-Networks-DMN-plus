from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time, json, warnings

from collections import deque
from .utils.generic_utils import Progbar


class CallbackList(object):
    def __init__(self, callbacks=[], queue_length=10):
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def _set_params(self, params):
        for callback in self.callbacks:
            callback._set_params(params)

    def _set_model(self, model):
        for callback in self.callbacks:
            callback._set_model(model)

    def on_epoch_begin(self, epoch, logs={}):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, epoch, logs={}):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1:
            warnings.warn('Method on_batch_begin() is slow compared '
                          'to the batch update (%f). Check your callbacks.' % delta_t_median)
        self._t_enter_batch = time.time()

    def on_batch_end(self, batch, logs={}):
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if self._delta_t_batch > 0. and delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1:
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f). Check your callbacks.' % delta_t_median)

    def on_train_begin(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):

    def __init__(self):
        pass

    def _set_params(self, params):
        self.params = params

    def _set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class BaseLogger(Callback):
    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('Epoch %d' % epoch)
            self.progbar = Progbar(target=self.params['nb_sample'],
                                   verbose=self.verbose)
        self.seen = 0
        self.totals = {}

    def on_batch_begin(self, batch, logs={}):
        if self.seen < self.params['nb_sample']:
            self.log_values = []

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # skip progbar update for the last batch; will be handled by on_epoch_end
        if self.verbose and self.seen < self.params['nb_sample']:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs={}):
        for k in self.params['metrics']:
            if k in self.totals:
                self.log_values.append((k, self.totals[k] / self.seen))
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)


class History(Callback):

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in self.totals.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v / self.seen)

        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, 
        append_epoch_name=False, save_every_X_epochs=1):

        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best = np.Inf
        self.append_epoch_name = append_epoch_name
        self.save_every_X_epochs = save_every_X_epochs

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.save_every_X_epochs != 0:
            return
        if self.append_epoch_name:
            import os
            file_name, file_extension = os.path.splitext(self.filepath)
            filepath = "%s_%05d%s" % (file_name, epoch, file_extension)
        else:
            filepath = self.filepath
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Can save best model only with %s available, skipping." % (self.monitor), RuntimeWarning)
            else:
                if current < self.best:
                    if self.verbose > 0:
                        print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                              % (epoch, self.monitor, self.best, current, filepath))
                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print("Epoch %05d: %s did not improve" % (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s" % (epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=0, verbose=0):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best = np.Inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % (self.monitor), RuntimeWarning)

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True
            self.wait += 1


class RemoteMonitor(Callback):
    def __init__(self, root='http://localhost:9000'):
        self.root = root

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs={}):
        import requests
        send = {}
        send['epoch'] = epoch

        for k, v in self.totals.items():
            send[k] = v / self.seen
        for k, v in logs.items():
            send[k] = v

        try:
            r = requests.post(self.root + '/publish/epoch/end/', {'data': json.dumps(send)})
        except:
            print('Warning: could not reach RemoteMonitor root server at ' + str(self.root))


class LearningRateScheduler(Callback):
    '''LearningRateScheduler
    schedule is a function that gets an epoch number as input and returns a new
    learning rate as output.
    '''
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs={}):
        model.lr.set_value(self.schedule(epoch))


class ModelTest(Callback):
    def __init__(self, Xt, Yt, T=10, test_every_X_epochs=1, batch_size=500, verbose=1, 
        loss=None, mean_y_train=None, std_y_train=None, tau=None):
        super(Callback, self).__init__()
        self.Xt = Xt
        self.Yt = np.array(Yt)
        self.T = T
        self.test_every_X_epochs = test_every_X_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.mean_y_train = mean_y_train
        self.std_y_train = std_y_train
        self.tau = tau

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.test_every_X_epochs != 0:
            return
        standard_prob = self.model.predict(
            self.Xt, batch_size=self.batch_size, verbose=self.verbose)
        MC_prob = np.array([self.model.predict_stochastic(
            self.Xt, batch_size=self.batch_size, verbose=self.verbose) 
                         for _ in xrange(self.T)])
        MC_prob_mean = np.mean(MC_prob, 0)
        # print(standard_prob * self.std_y_train + self.mean_y_train)
        # print(self.Yt)
        if self.loss == 'binary':
            standard_accuracy = np.mean(self.Yt == np.round(standard_prob.flatten()))
            MC_accuracy = np.mean(self.Yt == np.round(MC_prob_mean.flatten()))
        elif self.loss == 'categorical':
            standard_accuracy = np.mean(np.argmax(self.Yt, axis=-1) == np.argmax(standard_prob, axis=-1))
            MC_accuracy = np.mean(np.argmax(self.Yt, axis=-1) == np.argmax(MC_prob_mean, axis=-1))
        elif self.loss == 'euclidean':
            standard_prob = standard_prob * self.std_y_train + self.mean_y_train
            standard_accuracy = np.mean((self.Yt - standard_prob)**2.0, 0)**0.5
            MC_prob_mean = MC_prob_mean * self.std_y_train + self.mean_y_train
            MC_accuracy = np.mean((self.Yt - MC_prob_mean)**2.0, 0)**0.5
            Yt_hat = MC_prob * self.std_y_train + self.mean_y_train
            ll = (logsumexp(-0.5 * self.tau * (self.Yt[None] - Yt_hat)**2., 0) - np.log(self.T) 
                - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
            print("tau at epoch %05d: %0.5f" % (epoch, self.tau))
            print("ll at epoch %05d: %0.5f" % (epoch, np.mean(ll)))
        else:
            raise Exception('No loss: ' + loss)
        print(standard_accuracy)
        print("Standard accuracy/error at epoch %05d: %0.5f" % (epoch, float(standard_accuracy)))
        print("MC accuracy/error at epoch %05d: %0.5f" % (epoch, float(MC_accuracy)))
