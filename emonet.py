import argparse
import datetime

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, VGG19, ResNet50, Xception
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass


def model_architecture(args):
    net_model = VGG16(weights='imagenet', include_top=False,
                         input_shape=(args.input_height, args.input_width, args.channels_num))
    # net_model = ResNet50(weights='imagenet', include_top=False,
    #                      input_shape=(args.input_height, args.input_width, args.channels_num))
    # net_model = Xception(weights='imagenet', include_top=False, input_shape=(args.input_height,
    #                        args.input_width, args.channels_num))
    # net_model = VGG19(weights='imagenet', include_top=False,
    #                      input_shape=(args.input_height, args.input_width, args.channels_num))

    last_model_layer = net_model.output

    x = Flatten()(last_model_layer)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(args.classes_num, activation='softmax')(x)
    f_model = Model(net_model.input, preds)

    return f_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgsFormatter, description="Training script for EmotiW 2017.")

    parser.add_argument("--datapath", type=str, required=True, help="Path where Emotiw2017 data is stored.")
    parser.add_argument("--input_height", type=int, default=224, help="Input image height.")
    parser.add_argument("--input_width", type=int, default=224,  help="Input image width.")
    parser.add_argument("--channels_num", type=int, default=3, help="Input image channels number.")
    parser.add_argument("--classes_num", type=int, default=3, help="Classes number.")

    args = parser.parse_args()

    train_datagen = ImageDataGenerator(shear_range=0.2,
                                       rotation_range=10,
                                       zoom_range=0.1,
                                       horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        args.datapath + 'Train_crops/',
        target_size=(args.input_height, args.input_width),
        batch_size=16, class_mode='categorical')

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(
        args.datapath + 'Val_crops/',
        target_size=(args.input_height, args.input_width),
        batch_size=16, class_mode='categorical')

    model = model_architecture(args)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    tbCallBack = TensorBoard(log_dir='./logs/{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now()), histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=1600,
        nb_epoch=100,
        validation_data=test_generator,
        nb_val_samples=225,
        callbacks=[tbCallBack])

    model.save('emonet_{:%Y_%m_%d_%H_%M}.h5'.format(datetime.datetime.now()))