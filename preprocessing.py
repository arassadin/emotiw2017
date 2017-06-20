import dlib
import argparse
import os
from skimage import img_as_ubyte
import cv2


class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    pass


def process_data_subset(data_dir, dump_dir, classes, detector):
    # Create dirs for crops
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
        for c in classes:
            os.mkdir(dump_dir + c)

    total_faces_counter = 0

    print 'Processing ' + data_dir + ' subset.'

    for c in classes:
        files = os.listdir(data_dir + c)
        for f in files:
            filename = data_dir + c + '/' + f
            dump_filename = dump_dir + c + '/' + f
            print 'Current file to extract faces: {}'.format(filename)

            img = cv2.imread(filename)
            img = img_as_ubyte(img)
            img = img[:, :, 0:3]

            detections = detector(img, 1)

            for i, d in enumerate(detections):
                y1 = d.top()
                if y1 < 0:
                    y1 = 0
                y2 = d.bottom()
                if y2 > img.shape[0]:
                    y2 = img.shape[0]
                x1 = d.left()
                if x1 < 0:
                    x1 = 0
                x2 = d.right()
                if x2 > img.shape[1]:
                    x2 = img.shape[1]

                face = img[y1:y2, x1:x2, :]

                ind = dump_filename.rfind('.')
                to_dump = dump_filename[:ind] + '_' + str(i) + dump_filename[ind:]

                cv2.imwrite(to_dump, face)
                total_faces_counter += 1
    print 'Processing ' + data_dir + ' subset is done.'
    print 'Total faces extracted: ' + str(total_faces_counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgsFormatter, description="Preparation script for EmotiW 2017.")
    parser.add_argument("--datapath", type=str, required=True, help="Path where data is stored.")

    args = parser.parse_args()

    train_dir = args.datapath + 'Train/'
    val_dir = args.datapath + 'Val/'

    train_crops = args.datapath + 'Train_crops/'
    val_crops = args.datapath + 'Val_crops/'

    # Extract classes from train directory
    classes = os.listdir(train_dir)

    # Let's use face detector from dlib
    detector = dlib.get_frontal_face_detector()

    # Process train subset
    process_data_subset(train_dir, train_crops, classes, detector)

    # Process val subset
    process_data_subset(val_dir, val_crops, classes, detector)
