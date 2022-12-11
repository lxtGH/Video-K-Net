import os
import shutil

train_seqs = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
val_seqs = [2, 6, 7, 8, 10, 13, 14, 16, 18]
test_seqs = list(range(29))

# your download the KITTI STEP dataset.
data_root = os.path.expanduser('/data/data1/datasets/STEP/kitti/training/')
data_root_test = os.path.expanduser('/data/data1/datasets/STEP/kitti/testing/')
data_out = os.path.expanduser('/data/data1/datasets/STEP/kitti_out')


def build_panoptic(seq_id, input_dir, output_dir):
    input_panoptic_dir = os.path.join(input_dir, '{:04d}'.format(seq_id))
    print("Preparing seq id : {}".format(seq_id))
    panoptic_files = sorted(list(map(lambda x: str(x), os.listdir(input_panoptic_dir))))

    print("Dst dir is {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in panoptic_files:
        print(os.path.join(output_dir, '{:06d}_{}_panoptic.png'.format(seq_id, file.split('.')[0])))
        shutil.move(os.path.join(input_panoptic_dir, file),
                    os.path.join(output_dir, '{:06d}_{}_panoptic.png'.format(seq_id, file.split('.')[0])))


def build_img(seq_id, input_dir, output_dir):
    input_panoptic_dir = os.path.join(input_dir, '{:04d}'.format(seq_id))
    print("Preparing seq id : {}".format(seq_id))
    panoptic_files = sorted(list(map(lambda x: str(x), os.listdir(input_panoptic_dir))))

    print("Dst dir is {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in panoptic_files:
        print(os.path.join(output_dir, '{:06d}_{}_leftImg8bit.png'.format(seq_id, file.split('.')[0])))
        shutil.move(os.path.join(input_panoptic_dir, file),
                    os.path.join(output_dir, '{:06d}_{}_leftImg8bit.png'.format(seq_id, file.split('.')[0])))


if __name__ == '__main__':
    for seq_id in train_seqs:
        build_panoptic(seq_id, os.path.join(data_root, 'panoptic'), os.path.join(data_out, 'video_sequence', 'train'))

    for seq_id in val_seqs:
        build_panoptic(seq_id, os.path.join(data_root, 'panoptic'), os.path.join(data_out, 'video_sequence', 'val'))

    for seq_id in train_seqs:
        build_img(seq_id, os.path.join(data_root, 'image_02'), os.path.join(data_out, 'video_sequence', 'train'))

    for seq_id in val_seqs:
        build_img(seq_id, os.path.join(data_root, 'image_02'), os.path.join(data_out, 'video_sequence', 'val'))

    for seq_id in test_seqs:
        build_img(seq_id, os.path.join(data_root_test, 'image_02'), os.path.join(data_out, 'video_sequence', 'test'))