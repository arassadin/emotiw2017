import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adagrad, Adadelta
import numpy as np
from keras.callbacks import LearningRateScheduler, Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from time import time
from keras.utils.np_utils import to_categorical
import sys
import resnet
# from make_parallel import make_parallel

BASE_PATH = '/home/_datasets_/RaFD/'

model = None

class GlobalWotcher(Callback):

    def __init__(self, acc=False, react_time=600, dump_p=1000,
                 base_path='./'):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = None
        self.val_acc = None
        self.acc = acc
        if self.acc:
            self.train_acc = []
            self.val_acc = []
        self.react_time = react_time
        self.dump_period = dump_p
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self.cur_time = time()
        self.epoch += 1

        self.train_loss.append(logs.get('loss', None))
        self.val_loss.append(logs.get('val_loss', None))
        if self.acc:
            self.train_acc.append(logs.get('acc', None))
            self.val_acc.append(logs.get('val_acc', None))

        if self.cur_time - self.last_time >= self.react_time:
            self.dump_calculations()
            self.last_time = time()

        if self.epoch % self.dump_period == 0:
            self.dump_model()

    def on_train_begin(self, logs={}):
        self.last_time = time()
        self.epoch = 0

    def dump_model(self):
        model.save(os.path.join(self.base_path,
                                        'model_epoch={}.hdf5'.format(self.epoch)),
                           overwrite=True)
        sys.stdout.flush()
        print '\n######## Model Dumped ########'

    def dump_calculations(self):
        np.save(os.path.join(self.base_path, 'dump_train_loss'),
                np.asarray(self.train_loss))
        np.save(os.path.join(self.base_path, 'dump_val_loss'),
                np.asarray(self.val_loss))
        if self.acc:
            np.save(os.path.join(self.base_path, 'dump_train_acc'),
                    np.asarray(self.train_acc))
            np.save(os.path.join(self.base_path, 'dump_val_acc'),
                    np.asarray(self.val_acc))

    def on_train_end(self, logs={}):
        tstamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")

        model.save(os.path.join(self.base_path,
                                        'model_epoch={}.hdf5'.format(self.epoch)),
                           overwrite=True)

        plt.figure(figsize=(15, 10))
        plt.plot(xrange(1, len(self.train_loss) + 1), self.train_loss,
                 'r--', linewidth=3.0)
        plt.plot(xrange(1, len(self.train_loss) + 1), self.val_loss,
                 'g--', linewidth=3.0)
        plt.xlim(1, len(self.train_loss))
        plt.xlabel('epoch #')
        plt.ylabel('loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.base_path,
                                 './loss_train-val.png'))
        np.save(os.path.join(self.base_path, 'train_loss'),
                np.asarray(self.train_loss))
        np.save(os.path.join(self.base_path, 'val_loss'),
                np.asarray(self.val_loss))
        if self.acc:
            plt.figure(figsize=(15, 10))
            plt.plot(xrange(1, len(self.train_acc) + 1),
                     self.train_acc, 'r--', linewidth=3.0)
            plt.plot(xrange(1, len(self.val_acc) + 1),
                     self.val_acc, 'g--', linewidth=3.0)
            plt.xlim(1, len(self.train_acc))
            plt.xlabel('epoch #')
            plt.ylabel('accuracy')
            plt.grid(True)
            plt.savefig(os.path.join(self.base_path,
                                     './acc_train-val.png'))
            np.save(os.path.join(self.base_path, 'train_acc'),
                    np.asarray(self.train_acc))
            np.save(os.path.join(self.base_path, 'val_acc'),
                    np.asarray(self.val_acc))


BATCH_SIZE = 60
GPUS = [0, 1, 2, 3]

lr_base = 0.001
lr_curr = lr_base

opt = Adagrad(lr_base, decay=1e-4)
model = resnet.ResnetBuilder.build_resnet_101((3, 224, 224), 8)
# model.summary()
# model = make_parallel(model, gpus=GPUS)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

def normalize(arr):
    arr[:, 0] -= 64.1792
    arr[:, 0] /= 87.9963

    arr[:, 1] -= 49.1527
    arr[:, 1] /= 70.0328

    arr[:, 2] -= 40.8791
    arr[:, 2] /= 61.1681
    return arr

data_train = np.load(os.path.join(BASE_PATH, 'X_train.npy'))
data_train = normalize(data_train.astype(np.float32))
labels_train = to_categorical(np.load(os.path.join(BASE_PATH, 'y_train.npy')), num_classes=8)
print data_train.shape
print labels_train.shape

data_test = np.load(os.path.join(BASE_PATH, 'X_test.npy'))
data_test = normalize(data_test.astype(np.float32))
labels_test = to_categorical(np.load(os.path.join(BASE_PATH, 'y_test.npy')), num_classes=8)
print data_test.shape
print labels_test.shape

gen_train = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.1,
    channel_shift_range=5.0,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None)
# gen_train.fit(np.append(data_train, data_test, axis=0))

def epoch_sc(n):
    global lr_curr, lr_base
    lr_curr = lr_base * (1 - n / 1500.0)
    sys.stdout.flush()
    print '######## lr: {} ########'.format(lr_curr)
    return lr_curr

sc = LearningRateScheduler(epoch_sc)
gw = GlobalWotcher(acc=True, react_time=1000, dump_p=400,
                   base_path='./calc1')

model.fit_generator(gen_train.flow(data_train, labels_train), 
	steps_per_epoch=int(len(data_train) / float(BATCH_SIZE)),
# model.fit(data_train, labels_train, batch_size=BATCH_SIZE,
    epochs=2500,
    validation_data=(data_test, labels_test),
    callbacks=[gw])
