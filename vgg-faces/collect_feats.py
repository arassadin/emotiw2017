from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
import numpy as np

CLASS2LBL = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

def normalize(arr):
    arr = arr[:, :, ::-1].astype(np.float32)

    arr[0, :, :] -= 93.5940
    arr[1, :, :] -= 104.7624
    arr[2, :, :] -= 129.1863

    return arr

model = VGGFace(include_top=False, input_shape=(3, 224, 224), pooling='avg')

def routine(stat):
    X, y = [], []
    for fname, faces in stat.iteritems():
        _class = fname.split('/')[-2]
        lbl = CLASS2LBL[_class]
        y.append(lbl)

        faces = np.asarray([normalize(np.moveaxis(face, 2, 0)) for face in faces])
        preds = model.predict(faces)
        X.append(np.median(preds, axis=0))
        # break

    X = np.asarray(X)
    y = np.asarray(y, dtype=np.uint8)
    print X.shape, y.shape

    return X, y

faces_train = np.load('../faces_train.npy').item()
print 'Processing train...'
X_train, y_train = routine(faces_train)
np.save('X_train', X_train); np.save('y_train', y_train)

del X_train, y_train, faces_train

faces_val = np.load('../faces_val.npy').item()
print 'Processing val...'
X_val, y_val = routine(faces_val)
np.save('X_val', X_val); np.save('y_val', y_val)
