from __future__ import print_function

import argparse
import os
import os.path as osp
import torch.multiprocessing as multiprocessing
import numpy as np
import json
from PIL import Image
import pickle
from torch.utils.data import Dataset


class CityscapesVps(Dataset):

    def __init__(self):

        super(CityscapesVps, self).__init__()

        self.nframes_per_video = 6
        self.lambda_ = 5
        self.labeled_fid = 20

    def _save_image_single_core(self, proc_id, images_set, names_set, colors = None):

        def colorize(gray, palette):
            # gray: numpy array of the label and 1*3N size list palette
            color = Image.fromarray(gray.astype(np.uint8)).convert('P')
            color.putpalette(palette)
            return color

        for working_idx, (image, name) in enumerate(zip(images_set, names_set)):
            if colors is not None:
                image = colorize(image, colors)
            else:
                image = Image.fromarray(image)
            os.makedirs(os.path.dirname(name), exist_ok=True)
            image.save(name)

    def inference_panoptic_video(self, pred_pans_2ch, output_dir,
                                 categories,
                                 names,
                                 n_video=0):
        from panopticapi.utils import IdGenerator

        # Sample only frames with GT annotations.
        if len(pred_pans_2ch) != len(names):
            pred_pans_2ch = pred_pans_2ch[(self.labeled_fid // self.lambda_)::self.lambda_]
        categories = {el['id']: el for el in categories}
        color_generator = IdGenerator(categories)

        def get_pred_large(pan_2ch_all, vid_num, nframes_per_video=6):
            vid_num = len(pan_2ch_all) // nframes_per_video  # 10
            cpu_num = multiprocessing.cpu_count() // 2  # 32 --> 16
            nprocs = min(vid_num, cpu_num)  # 10
            max_nframes = cpu_num * nframes_per_video
            nsplits = (len(pan_2ch_all) - 1) // max_nframes + 1
            annotations, pan_all = [], []
            for i in range(0, len(pan_2ch_all), max_nframes):
                print('==> Read and convert VPS output - split %d/%d' % ((i // max_nframes) + 1, nsplits))
                pan_2ch_part = pan_2ch_all[i:min(
                    i + max_nframes, len(pan_2ch_all))]
                pan_2ch_split = np.array_split(pan_2ch_part, nprocs)
                workers = multiprocessing.Pool(processes=nprocs)
                processes = []
                for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
                    p = workers.apply_async(
                        self.converter_2ch_track_core,
                        (proc_id, pan_2ch_set, color_generator))
                    processes.append(p)
                workers.close()
                workers.join()

                for p in processes:
                    p = p.get()
                    annotations.extend(p[0])
                    pan_all.extend(p[1])

            pan_json = {'annotations': annotations}
            return pan_all, pan_json

        def save_image(images, save_folder, names, colors=None):
            os.makedirs(save_folder, exist_ok=True)

            names = [osp.join(save_folder,
                              name.replace('_leftImg8bit', '').replace('_newImg8bit', '').replace('jpg', 'png').replace(
                                  'jpeg', 'png')) for name in names]
            cpu_num = multiprocessing.cpu_count() // 2
            images_split = np.array_split(images, cpu_num)
            names_split = np.array_split(names, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            for proc_id, (images_set, names_set) in enumerate(zip(images_split, names_split)):
                workers.apply_async(self._save_image_single_core, (proc_id, images_set, names_set, colors))
            workers.close()
            workers.join()

        # inference_panoptic_video
        pred_pans, pred_json = get_pred_large(pred_pans_2ch,
                                              vid_num=n_video)
        print('--------------------------------------')
        print('==> Saving VPS output png files')
        os.makedirs(output_dir, exist_ok=True)
        save_image(pred_pans_2ch, osp.join(output_dir, 'pan_2ch'), names)
        save_image(pred_pans, osp.join(output_dir, 'pan_pred'), names)
        print('==> Saving pred.jsons file')
        json.dump(pred_json, open(osp.join(output_dir, 'pred.json'), 'w'))
        print('--------------------------------------')

        return pred_pans, pred_json

    def converter_2ch_track_core(self, proc_id, pan_2ch_set, color_generator):
        from panopticapi.utils import rgb2id

        OFFSET = 1000
        VOID = 255
        annotations, pan_all = [], []
        # reference dict to used color
        inst2color = {}
        for idx in range(len(pan_2ch_set)):
            pan_2ch = np.uint32(pan_2ch_set[idx])
            # pan_2ch: ss-seg maps[:,:,0], id-seg maps[:,:,1]
            pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 1]

            pan_format = np.zeros((pan_2ch.shape[0], pan_2ch.shape[1], 3), dtype=np.uint8)
            l = np.unique(pan)

            segm_info = {}
            for el in l:
                sem = el // OFFSET

                if sem == VOID:
                    continue
                mask = pan == el
                #### handling used color for inst id
                if el % OFFSET > 0:
                    # if el > OFFSET:
                    # things class
                    if el in inst2color:
                        color = inst2color[el]
                    else:
                        color = color_generator.get_color(sem)
                        inst2color[el] = color
                else:
                    # stuff class
                    color = color_generator.get_color(sem)

                pan_format[mask] = color
                index = np.where(mask)
                x = index[1].min()
                y = index[0].min()
                width = index[1].max() - x
                height = index[0].max() - y

                dt = {"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)),
                      "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()}
                segment_id = int(rgb2id(color))
                segm_info[segment_id] = dt

            # annotations.append({"segments_info": segm_info})
            pan_all.append(pan_format)

            gt_pan = np.uint32(pan_format)
            # rgb2id for evaluation
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            labels, labels_cnt = np.unique(pan_gt, return_counts=True)
            for label, area in zip(labels, labels_cnt):
                if label == 0:
                    continue
                if label not in segm_info.keys():
                    print('label:', label)
                    raise KeyError('label not in segm_info keys.')

                segm_info[label]["area"] = int(area)
            segm_info = [v for k, v in segm_info.items()]

            annotations.append({"segments_info": segm_info})

        return annotations, pan_all