import os
import random
from typing import Dict, List

import copy

import mmcv
import numpy as np
import torch

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import CustomDataset
from mmdet.utils import get_root_logger

from external.dataset.mIoU import eval_miou


class SeqObj:
    # This divisor is orthogonal with panoptic class-instance divisor.
    DIVISOR = 1000000

    def __init__(self, the_dict: Dict):
        self.dict = the_dict
        assert 'seq_id' in self.dict and 'img_id' in self.dict

    def __hash__(self):
        return self.dict['seq_id'] * self.DIVISOR + self.dict['img_id']

    def __eq__(self, other):
        return self.dict['seq_id'] == other.dict['seq_id'] and self.dict['img_id'] == other.dict['img_id']

    def __getitem__(self, attr):
        return self.dict[attr]


@DATASETS.register_module()
class KITTISTEPDVPSDataset:
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    def __init__(self,
                 pipeline=None,
                 data_root=None,
                 test_mode=False,
                 split='train',
                 ref_seq_index: List[int] = None,
                 is_instance_only: bool = True,
                 with_depth: bool = False
                 ):
        assert data_root is not None
        data_root = os.path.expanduser(data_root)
        video_seq_dir = os.path.join(data_root, 'video_sequence', split)
        assert os.path.exists(video_seq_dir)
        assert 'leftImg8bit' not in video_seq_dir

        self.num_thing_classes = 2
        self.num_stuff_classes = 17
        self.thing_before_stuff = False

        # ref_seq_index is None means no ref img
        if ref_seq_index is None:
            ref_seq_index = []

        filenames = list(map(lambda x: str(x), os.listdir(video_seq_dir)))
        img_names = sorted(list(filter(lambda x: 'leftImg8bit' in x, filenames)))

        images = []
        for item in img_names:
            seq_id, img_id, _ = item.split(sep="_", maxsplit=2)
            if int(seq_id) == 1 and int(img_id) in [177, 178, 179, 180] and with_depth:
                continue
            item_full = os.path.join(video_seq_dir, item)
            images.append(SeqObj({
                'seq_id': int(seq_id),
                'img_id': int(img_id),
                'img': item_full,
                'depth': item_full.replace('leftImg8bit', 'depth') if with_depth else None,
                'ann': item_full.replace('leftImg8bit', 'panoptic'),
                # This should be modified carefully for each dataset. Usually 255.
                'no_obj_class': 255
            }))
            assert os.path.exists(images[-1]['img'])
            assert images[-1]['depth'] is None or os.path.exists(images[-1]['depth']), \
                "Missing depth : {}".format(images[-1]['depth'])
            # assert os.path.exists(images[-1]['ann'])

        reference_images = {hash(image): image for image in images}
        sequences = []
        for img_cur in images:
            is_seq = True
            seq_now = [img_cur.dict]
            if ref_seq_index:
                for index in random.choices(ref_seq_index, k=1):
                    query_obj = SeqObj({
                        'seq_id': img_cur.dict['seq_id'],
                        'img_id': img_cur.dict['img_id'] + index
                    })
                    if hash(query_obj) in reference_images:
                        seq_now.append(reference_images[hash(query_obj)].dict)
                    else:
                        is_seq = False
                        break
            if is_seq:
                sequences.append(seq_now)

        self.sequences = sequences
        self.ref_seq_index = ref_seq_index

        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()
        self.is_instance_only = is_instance_only

        # For evaluation
        self.max_ins = 10000
        self.no_obj_id = 255

    def pre_pipelines(self, results):
        for _results in results:
            _results['img_info'] = []
            _results['thing_lower'] = 0 if self.thing_before_stuff else self.num_stuff_classes
            _results['thing_upper'] = self.num_thing_classes \
                if self.thing_before_stuff else self.num_stuff_classes + self.num_thing_classes
            _results['is_instance_only'] = self.is_instance_only
            _results['ori_filename'] = os.path.basename(_results['img'])

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        results = copy.deepcopy(self.sequences[idx])
        self.pre_pipelines(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.sequences[idx])
        self.pre_pipelines(results)
        # During test time, one image inference does not requires seq
        if not self.ref_seq_index:
            results = results[0]
        return self.pipeline(results)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

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

    def __len__(self):
        """Total number of samples of data."""
        return len(self.sequences)

    def _set_groups(self):
        return np.zeros((len(self)), dtype=np.int64)

    # The evaluate func
    def evaluate(
            self,
            results,
            **kwargs
    ):
        # logger and metric
        thing_knet2real = [11, 13]
        pred_results_handled = []
        pred_depth = []
        pred_depth_final = []
        item_id = 0
        sem_preds = []
        for item in results:
            if item[-1] is not None:
                # With depth
                bbox_results, mask_results, seg_results, depth, depth_final = item
                pred_depth.append(depth)
                pred_depth_final.append(depth_final)
            else:
                bbox_results, mask_results, seg_results, _, _ = item
            # in seg_info id starts from 1
            inst_map, seg_info = seg_results
            cat_map = np.zeros_like(inst_map) + self.num_thing_classes + self.num_stuff_classes
            for instance in seg_info:
                cat_cur = instance['category_id']
                if instance['isthing']:
                    cat_cur = thing_knet2real[cat_cur]
                else:
                    if self.thing_before_stuff:
                        raise NotImplementedError
                    else:
                        # stuff starts from 1 in the model
                        cat_cur -= 1
                        offset = 0
                        for thing_id in thing_knet2real:
                            if cat_cur + offset >= thing_id:
                                offset += 1
                        cat_cur += offset
                assert cat_cur < self.num_thing_classes + self.num_stuff_classes
                cat_map[inst_map == instance['id']] = cat_cur
                if not instance['isthing']:
                    inst_map[inst_map == instance['id']] = 0
            pred_results_handled.append(cat_map.astype(np.int32) * self.max_ins + inst_map.astype(np.int32))
            item_id += 1
            sem_preds.append(cat_map)

        gt_panseg = []
        gt_depth = []
        sem_targets = []
        for item in self.sequences:
            # Only for single
            item = item[0]
            # Only for single
            id_map = mmcv.imread(item['ann'], flag='color', channel_order='rgb')
            gt_semantic_seg = id_map[..., 0].astype(np.int32)
            sem_targets.append(gt_semantic_seg)
            gt_inst_map = id_map[..., 1].astype(np.int32) * 256 + id_map[..., 2].astype(np.int32)
            ps_id = gt_semantic_seg * self.max_ins + gt_inst_map
            gt_panseg.append(ps_id)
            if len(pred_depth) > 0:
                gt_depth_cur = mmcv.imread(item['depth'], flag='unchanged').astype(np.float32) / 256.
                gt_depth.append(gt_depth_cur)

        vpq_results = []
        for pred, gt in zip(pred_results_handled, gt_panseg):
            vpq_result = vpq_eval([pred, gt])
            vpq_results.append(vpq_result)

        iou_per_class = np.stack([result[0] for result in vpq_results]).sum(axis=0)[
                        :self.num_thing_classes + self.num_stuff_classes]
        tp_per_class = np.stack([result[1] for result in vpq_results]).sum(axis=0)[
                       :self.num_thing_classes + self.num_stuff_classes]
        fn_per_class = np.stack([result[2] for result in vpq_results]).sum(axis=0)[
                       :self.num_thing_classes + self.num_stuff_classes]
        fp_per_class = np.stack([result[3] for result in vpq_results]).sum(axis=0)[
                       :self.num_thing_classes + self.num_stuff_classes]

        abs_rels = []
        abs_rel_finals = []
        if len(pred_depth) > 0:
            for pred, pred_final, gt in zip(pred_depth, pred_depth_final, gt_depth):
                depth_mask = gt > 0.
                abs_rel_normal = np.mean(
                    np.abs(
                        pred[depth_mask] -
                        gt[depth_mask]) /
                    gt[depth_mask])
                abs_rel_final = np.mean(
                    np.abs(
                        pred_final[depth_mask] -
                        gt[depth_mask]) /
                    gt[depth_mask])
                abs_rels.append(abs_rel_normal)
                abs_rel_finals.append(abs_rel_final)
            abs_rel = np.stack(abs_rels).mean(axis=0)
            abs_rel_final = np.stack(abs_rel_finals).mean(axis=0)
        else:
            abs_rel = 0.
            abs_rel_final = 0.

        # calculate the PQs
        epsilon = 0.
        sq = iou_per_class / (tp_per_class + epsilon)
        rq = tp_per_class / (tp_per_class + 0.5 *
                             fn_per_class + 0.5 * fp_per_class + epsilon)
        pq = sq * rq
        things_index = np.zeros((19,)).astype(bool)
        things_index[11] = True
        things_index[13] = True
        stuff_pq = pq[np.logical_not(things_index)]
        things_pq = pq[things_index]

        miou_per_class = eval_miou(sem_preds, sem_targets, num_classes=self.num_thing_classes + self.num_stuff_classes)
        print("class        pq\t\tsq\t\trq\t\ttp\t\tfp\t\tfn\t\tmIoU")

        for i in range(len(self.CLASSES)):
            print("{}{}{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.0f}\t\t{:.0f}\t\t{:.0f}\t\t{:.3f}".format(
                self.CLASSES[i], ' ' * (13 - len(self.CLASSES[i])), pq[i], sq[i], rq[i], tp_per_class[i],
                fp_per_class[i], fn_per_class[i], miou_per_class[i]
            ))

        return {
            "abs_rel": abs_rel,
            "abs_rel_final": abs_rel_final,
            "PQ": np.nan_to_num(pq).mean() * 100,
            "Stuff PQ": np.nan_to_num(stuff_pq).mean() * 100,
            "Things PQ": np.nan_to_num(things_pq).mean() * 100,
            "mIoU": np.nan_to_num(miou_per_class).mean() * 100,
        }


def vpq_eval(element):
    import six
    pred_ids, gt_ids = element
    max_ins = 10000
    ign_id = 255
    offset = 2 ** 30
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
    import dataset.dvps_pipelines.loading
    import dataset.dvps_pipelines.transforms
    import dataset.pipelines.transforms
    import dataset.pipelines.formatting

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

    test_pipeline = [
        dict(type='LoadMultiImagesDirect'),
        dict(type='SeqPadWithDepth', size_divisor=32),
        dict(type='SeqNormalize', **img_norm_cfg),
        dict(
            type='VideoCollect',
            keys=['img']),
        dict(type='ConcatVideoReferences'),
        dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
    ]

    data = KITTISTEPDVPSDataset(
        pipeline=[
            dict(type='LoadMultiImagesDirect'),
            dict(type='LoadMultiAnnotationsDirect', with_depth=True, divisor=-1),
            dict(type='SeqFlipWithDepth', flip_ratio=0.5),
            dict(type='SeqPadWithDepth', size_divisor=32),
            dict(type='SeqNormalize', **img_norm_cfg),
            dict(
                type='VideoCollect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_depth']),
            dict(type='ConcatVideoReferences'),
            dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
        ],
        data_root=os.path.expanduser('~/datasets/kitti-step'),
        split='val',
        ref_seq_index=[-1, 1],
        with_depth=True,
    )
    np.set_string_function(lambda x: '<{} ; {}>'.format(x.shape, x.dtype))
    torch.set_printoptions(profile='short')
    for item in data:
        print(item)
