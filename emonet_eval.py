import argparse
import os
import random
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass


def aggregate_max_occurrence(labs):
    return max(labs, key=labs.count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgsFormatter, description="Evaluation script for EmotiW 2017 val.")

    parser.add_argument("--datapath", type=str, required=True, help="Path where Emotiw2017 data is stored.")
    parser.add_argument("--input_height", type=int, default=224, help="Input image height.")
    parser.add_argument("--input_width", type=int, default=224,  help="Input image width.")
    parser.add_argument("--channels_num", type=int, default=3, help="Input image channels number.")
    parser.add_argument("--classes_num", type=int, default=3, help="Classes number.")
    parser.add_argument("--snapshot_name", type=str, required=True, help="Path to snapshot to evaluate.")

    args = parser.parse_args()

    model = load_model(args.snapshot_name)
    model.summary()

    classes = os.listdir(args.datapath + 'Val_crops/')
    full_classes = os.listdir(args.datapath + 'Val/')

    total_acc = 0.0
    counter = 0
    total_counter = 0
    per_class_accs = [0.0, 0.0, 0.0]

    for i in range(len(full_classes)):
        c = full_classes[i]
        current_acc = 0
        files = os.listdir(args.datapath + 'Val/' + c)
        files = sorted(files)

        total_counter += len(files)

        cropped_files = os.listdir(args.datapath + 'Val_crops/' + c)
        cropped_files = sorted(cropped_files)

        for f in files:
            counter += 1
            ind = f.rfind('.')
            output_names = [name for name in cropped_files if (f[:ind] in name)]

            labels = []
            if len(output_names) == 0:
                # if we don't have any faces let's randomize :)
                labels.append(random.randint(0, 2))
            else:
                for name in output_names:
                    img = load_img(args.datapath + 'Val_crops/' + c + '/' + name, target_size=(224, 224))
                    X = img_to_array(img, data_format='channels_last')
                    X = X.reshape((1, 224, 224, 3))
                    out = model.predict(X)
                    labels.append(np.argmax(out))
            print labels

            # Simple aggregation
            pr = aggregate_max_occurrence(labs=labels)

            print 'Predicted label: ' + str(pr)
            print 'Ground truth: ' + str(i)

            if pr == i:
                total_acc += 1
                current_acc += 1
            if counter % 100 == 0:
                print 'Processed: ' + str(counter)
        per_class_accs[i] = float(current_acc)/len(files)
    total_acc *= 1.0 / total_counter
    print '------------------------------------------'
    print 'Accuracy: ' + str(total_acc)
    print 'Per-class accuracy: ' + str(per_class_accs)
    print '------------------------------------------'