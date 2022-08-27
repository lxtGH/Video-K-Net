import os

import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines.compose import Compose

from external.dataset.mIoU import eval_miou


@DATASETS.register_module()
class CityscapesSTEP:
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    def __init__(
            self,
            pipeline=None,
            data_root=None,
            test_mode=False,
            split='train',
    ):
        # Let's figure out where is the cityscapes first
        assert os.path.exists(os.path.join(data_root, 'license.txt')), \
            "It seems that '{}' is not the root folder of cityscapes".format(data_root)
        assert os.path.exists(os.path.join(data_root, 'leftImg8bit')), \
            "leftImg8bit cannot be found."
        assert os.path.exists(os.path.join(data_root, 'gtFine')), \
            "gtFine cannot be found."

        if pipeline is None:
            pipeline = []

        image_main_dir = os.path.join(data_root, 'leftImg8bit', split)
        gt_dir = os.path.join(data_root, 'gtFine', split)

        locations = os.listdir(image_main_dir)
        samples = []
        for loc in locations:
            for sample in os.listdir(os.path.join(image_main_dir, loc)):
                location, seq_id, img_id, _ = sample.split('_')
                assert location == loc
                samples.append((location, int(seq_id), int(img_id)))
        samples = sorted(samples)
        self.samples = samples

        # Set the image dirs
        self.gt_dir = gt_dir
        self.img_dir = image_main_dir

        self.pipeline = Compose(pipeline)
        self.load_ann_pipeline = Compose([
            dict(
                type='LoadAnnotationsInstanceMasks',
                with_mask=False,
                with_seg=True,
                with_inst=True,
            ),
        ])
        self.test_mode = test_mode

        self.flag = self._set_groups()

        # eval
        self.max_ins = 1000
        self.no_obj_id = 255

    def pre_pipeline(self, results):
        results['img_prefix'] = None
        results['img_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['bbox_fields'] = []
        return results

    def prepare_test_img(self, idx):
        get_idx = self.samples[idx]
        filename = os.path.join(self.img_dir, get_idx[0], '{}_{:06d}_{:06d}_leftImg8bit.png'.format(*get_idx))
        results = {
            'img_info': {
                'filename': filename
            }
        }
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_val_annotation(self, idx):
        get_idx = self.samples[idx]
        results = {
            'ann_info': {
                'seg_map': os.path.join(self.gt_dir, get_idx[0],
                                        '{}_{:06d}_{:06d}_gtFine_labelTrainIds.png'.format(*get_idx)),
                'inst_map': os.path.join(self.gt_dir, get_idx[0],
                                         '{}_{:06d}_{:06d}_gtFine_instanceTrainIds.png'.format(*get_idx)),
            }
        }
        results = self.pre_pipeline(results)
        return self.load_ann_pipeline(results)

    def prepare_train_img(self, idx):
        get_idx = self.samples[idx]
        filename = os.path.join(self.img_dir, get_idx[0], '{}_{:06d}_{:06d}_leftImg8bit.png'.format(*get_idx))
        results = {
            'img_info': {
                'filename': filename
            },
            'ann_info': {
                'seg_map': os.path.join(self.gt_dir, get_idx[0],
                                        '{}_{:06d}_{:06d}_gtFine_labelTrainIds.png'.format(*get_idx)),
                'inst_map': os.path.join(self.gt_dir, get_idx[0],
                                         '{}_{:06d}_{:06d}_gtFine_instanceTrainIds.png'.format(*get_idx)),
            }
        }
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    # Copy and Modify from mmdet
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            while True:
                cur_data = self.prepare_train_img(idx)
                if cur_data is None:
                    idx = self._rand_another(idx)
                    continue
                return cur_data

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.arange(len(self))
        return np.random.choice(pool)

    def __len__(self):
        return len(self.samples)

    def _set_groups(self):
        return np.zeros((len(self)), dtype=np.int64)

    # The evaluate func
    def evaluate(
            self,
            results,
            **kwargs
    ):
        # logger and metric
        thing_lower = 11
        thing_upper = 19

        num_thing_classes = 8
        num_stuff_classes = 11
        pred_results_handled = []
        sem_preds = []

        thing_knet2real = [11, 13]

        for item in results:
            bbox_results, mask_results, seg_results, _, _ = item
            # in seg_info id starts from 1
            inst_map, seg_info = seg_results
            cat_map = np.zeros_like(inst_map) + num_thing_classes + num_stuff_classes
            for instance in seg_info:
                cat_cur = instance['category_id']
                if instance['isthing']:
                    cat_cur = thing_knet2real[cat_cur]
                else:
                    cat_cur -= 1
                    offset = 0
                    for thing_id in thing_knet2real:
                        if cat_cur + offset >= thing_id:
                            offset += 1
                    cat_cur += offset
                assert cat_cur < num_thing_classes + num_stuff_classes
                cat_map[inst_map == instance['id']] = cat_cur
                if not instance['isthing']:
                    inst_map[inst_map == instance['id']] = 0
            pred_results_handled.append(cat_map.astype(np.int32) * self.max_ins + inst_map.astype(np.int32))
            sem_preds.append(cat_map)

        gt_panseg = []
        sem_targets = []
        for idx in range(len(self)):
            results = self.prepare_val_annotation(idx)
            panseg_map = results['gt_instance_map']
            sem_targets.append(panseg_map // self.max_ins)
            gt_panseg.append(panseg_map)

        vpq_results = []
        for pred, gt in zip(pred_results_handled, gt_panseg):
            vpq_result = vpq_eval([pred, gt])
            vpq_results.append(vpq_result)

        iou_per_class = np.stack([result[0] for result in vpq_results]).sum(axis=0)[
                        :num_thing_classes + num_stuff_classes]
        tp_per_class = np.stack([result[1] for result in vpq_results]).sum(axis=0)[
                       :num_thing_classes + num_stuff_classes]
        fn_per_class = np.stack([result[2] for result in vpq_results]).sum(axis=0)[
                       :num_thing_classes + num_stuff_classes]
        fp_per_class = np.stack([result[3] for result in vpq_results]).sum(axis=0)[
                       :num_thing_classes + num_stuff_classes]

        # calculate the PQs
        epsilon = 0.
        sq = iou_per_class / (tp_per_class + epsilon)
        rq = tp_per_class / (tp_per_class + 0.5 *
                             fn_per_class + 0.5 * fp_per_class + epsilon)
        pq = sq * rq
        # stuff_pq = pq[:num_stuff_classes]
        # things_pq = pq[num_stuff_classes:]
        things_index = np.zeros((19,)).astype(bool)
        things_index[11] = True
        things_index[13] = True
        stuff_pq = pq[np.logical_not(things_index)]
        things_pq = pq[things_index]

        miou_per_class = eval_miou(sem_preds, sem_targets, num_classes=num_thing_classes + num_stuff_classes)

        pq = sq * rq
        print("class        pq\t\tsq\t\trq\t\ttp\t\tfp\t\tfn\t\tmIoU")

        for i in range(len(self.CLASSES)):
            print("{}{}{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.0f}\t\t{:.0f}\t\t{:.0f}\t\t{:.3f}".format(
                self.CLASSES[i], ' '*(13 - len(self.CLASSES[i])), pq[i], sq[i], rq[i], tp_per_class[i],
                fp_per_class[i], fn_per_class[i], miou_per_class[i]
            ))

        return {
            "PQ": np.nan_to_num(pq).mean() * 100,
            "Stuff PQ": np.nan_to_num(stuff_pq).mean() * 100,
            "Things PQ": np.nan_to_num(things_pq).mean() * 100,
            "mIoU":np.nan_to_num(miou_per_class).mean() * 100,
        }


def vpq_eval(element):
    import six
    pred_ids, gt_ids = element
    max_ins = 1000
    ign_id = 255
    offset = 256 * 256
    num_cat = 19 + 1

    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    def _ids_to_counts(id_array):
        ids, counts = np.unique(id_array, return_counts=True)
        return dict(six.moves.zip(ids, counts))

    pred_areas = _ids_to_counts(pred_ids)
    gt_areas = _ids_to_counts(gt_ids)

    void_id = ign_id * max_ins
    ign_ids = {
        gt_id for gt_id in six.iterkeys(gt_areas)
        if (gt_id // max_ins) == ign_id
    }

    int_ids = gt_ids.astype(np.int64) * offset + pred_ids.astype(np.int64)
    int_areas = _ids_to_counts(int_ids)

    def prediction_void_overlap(pred_id):
        void_int_id = void_id * offset + pred_id
        return int_areas.get(void_int_id, 0)

    def prediction_ignored_overlap(pred_id):
        total_ignored_overlap = 0
        for _ign_id in ign_ids:
            int_id = _ign_id * offset + pred_id
            total_ignored_overlap += int_areas.get(int_id, 0)
        return total_ignored_overlap

    gt_matched = set()
    pred_matched = set()

    for int_id, int_area in six.iteritems(int_areas):
        gt_id = int(int_id // offset)
        gt_cat = int(gt_id // max_ins)
        pred_id = int(int_id % offset)
        pred_cat = int(pred_id // max_ins)
        if gt_cat != pred_cat:
            continue
        union = (
                gt_areas[gt_id] + pred_areas[pred_id] - int_area -
                prediction_void_overlap(pred_id)
        )
        iou = int_area / union
        if iou > 0.5:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in six.iterkeys(gt_areas):
        if gt_id in gt_matched:
            continue
        cat_id = gt_id // max_ins
        if cat_id == ign_id:
            continue
        fn_per_class[cat_id] += 1

    for pred_id in six.iterkeys(pred_areas):
        if pred_id in pred_matched:
            continue
        if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
            continue
        cat = pred_id // max_ins
        fp_per_class[cat] += 1

    return iou_per_class, tp_per_class, fn_per_class, fp_per_class


if __name__ == '__main__':
    import dataset.pipelines.loading
    import dataset.pipelines.transforms

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    )
    train_pipelines = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotationsInstanceMasks', cherry=[11, 13]),
        dict(type='KNetInsAdapterCherryPick', stuff_nums=11, cherry=[11, 13]),
        dict(type='Resize', img_scale=(1024, 2048), ratio_range=[0.5, 2.0], keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='RandomCrop', crop_size=(1024, 2048)),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='PadFutureMMDet', size_divisor=32, pad_val=dict(img=0, masks=0, seg=255)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_masks', 'gt_labels', 'gt_semantic_seg'],
             meta_keys=('ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                        'flip_direction', 'img_norm_cfg')
             ),
    ]
    data = CityscapesSTEP(
        pipeline=train_pipelines,
        data_root='data/cityscapes',
        split='train',
        test_mode=False
    )
    for item in data:
        print(item)
