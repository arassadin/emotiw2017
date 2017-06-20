import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import datetime


class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgsFormatter, description="Training script for EmotiW 2017.")
    parser.add_argument("--datapath", type=str, required=True, help="Path where Emotiw2017 data is stored.")
    parser.add_argument("--input_height", type=int, default=224, help="Input image height.")
    parser.add_argument("--input_width", type=int, default=224,  help="Input image width")

    args = parser.parse_args()

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       rotation_range=10,
                                       zoom_range=0.1,
                                       horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        args.datapath + 'Train_crops/',
        target_size=(args.input_height, args.input_width),
        batch_size=32, class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        args.datapath + 'Val_crops/',
        target_size=(args.input_height, args.input_width),
        batch_size=32, class_mode='categorical')

    model = VGG16(weights='imagenet', include_top=False, input_shape=(args.input_height, args.input_width, 3))

    last = model.output
    x = Flatten()(last)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(3, activation='softmax')(x)

    f_model = Model(model.input, preds)

    f_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    tbCallBack = TensorBoard(log_dir='./logs/{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now()), histogram_freq=0, write_graph=True, write_images=True)

    history = f_model.fit_generator(
        train_generator,
        samples_per_epoch=400,
        nb_epoch=5,
        validation_data=test_generator,
        nb_val_samples=225,
        callbacks=[tbCallBack])

    f_model.save_weights('model.h5')