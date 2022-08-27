import os
from typing import Dict, List

import copy

import mmcv
import numpy as np
import random
import torch

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose


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
class KITTIDVPSDataset:
    CLASSES = (
        'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist'
    )

    def __init__(self,
                 pipeline=None,
                 data_root=None,
                 test_mode=False,
                 split='train',
                 ref_seq_index: List[int] = None,
                 is_instance_only: bool = True,
                 ):
        assert data_root is not None
        data_root = os.path.expanduser(data_root)
        video_seq_dir = os.path.join(data_root, 'video_sequence', split)
        assert os.path.exists(video_seq_dir)
        assert 'leftImg8bit' not in video_seq_dir

        self.num_thing_classes = 8
        self.num_stuff_classes = 11
        self.thing_before_stuff = True

        # ref_seq_index is None means no ref img
        if ref_seq_index is None:
            ref_seq_index = []

        filenames = list(map(lambda x: str(x), os.listdir(video_seq_dir)))
        depth_names = sorted(list(filter(lambda x: 'depth' in x, filenames)))
        # No depth annotation
        if not depth_names:
            depth_names = sorted(list(filter(lambda x: 'leftImg8bit' in x, filenames)))

        images = []
        for item in depth_names:
            seq_id, img_id, _ = item.split(sep="_", maxsplit=2)
            item_full = os.path.join(video_seq_dir, item)
            images.append(SeqObj({
                'seq_id': int(seq_id),
                'img_id': int(img_id),
                'img': os.path.join(video_seq_dir, "{}_{}_{}.png".format(seq_id, img_id, 'leftImg8bit')),
                'depth': item_full,
                'ann_class': os.path.join(video_seq_dir, "{}_{}_{}.png".format(seq_id, img_id, 'gtFine_class')),
                'ann_inst': os.path.join(video_seq_dir, "{}_{}_{}.png".format(seq_id, img_id, 'gtFine_instance')),
                # This should be modified carefully for each dataset. Usually 255.
                'no_obj_class': 255
            }))
            assert os.path.exists(images[-1]['img'])
            if not test_mode:
                assert os.path.exists(images[-1]['depth'])
                assert os.path.exists(images[-1]['ann_class'])
                assert os.path.exists(images[-1]['ann_inst'])

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
        self.max_ins = 1000
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
        thing_lower = 0 if self.thing_before_stuff else self.num_stuff_classes
        thing_upper = self.num_thing_classes \
            if self.thing_before_stuff else self.num_stuff_classes + self.num_thing_classes
        pred_results_handled = []
        pred_depth = []
        pred_depth_final = []
        for item in results:
            bbox_results, mask_results, seg_results, depth, depth_final = item
            pred_depth.append(depth)
            pred_depth_final.append(depth_final)
            # in seg_info id starts from 1
            inst_map, seg_info = seg_results
            cat_map = np.zeros_like(inst_map) + self.num_thing_classes + self.num_stuff_classes
            for instance in seg_info:
                cat_cur = instance['category_id']
                if instance['isthing']:
                    cat_cur += thing_lower
                else:
                    if self.thing_before_stuff:
                        cat_cur = cat_cur - 1 + thing_upper
                    else:
                        # stuff starts from 1 in the model
                        cat_cur -= 1
                assert cat_cur < self.num_thing_classes + self.num_stuff_classes
                cat_map[inst_map == instance['id']] = cat_cur
                if not instance['isthing']:
                    inst_map[inst_map == instance['id']] = 0
            pred_results_handled.append(cat_map.astype(np.int32) * 10000 + inst_map.astype(np.int32))

        gt_panseg = []
        gt_depth = []
        for item in self.sequences:
            # Only for single
            item = item[0]
            # Only for single
            cat_id = mmcv.imread(item['ann_class'], flag='unchanged').astype(np.int32)
            inst_id = mmcv.imread(item['ann_inst'], flag='unchanged').astype(np.int32)
            ps_id = cat_id * 10000 + inst_id
            gt_panseg.append(ps_id)
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

        # calculate the PQs
        epsilon = 0.
        sq = iou_per_class / (tp_per_class + epsilon)
        rq = tp_per_class / (tp_per_class + 0.5 *
                             fn_per_class + 0.5 * fp_per_class + epsilon)
        print("tp per class")
        print(tp_per_class)
        print("fp per class")
        print(fp_per_class)
        print("fn per class")
        print(fn_per_class)

        pq = sq * rq
        print("PQ")
        print(pq[:thing_upper])
        print(pq[thing_upper:])
        print("SQ")
        print(sq)
        print("RQ")
        print(rq)
        stuff_pq = pq[:thing_upper]
        things_pq = pq[thing_upper:]

        return {
            "abs_rel": abs_rel,
            "abs_rel_final": abs_rel_final,
            "PQ": np.nan_to_num(pq).mean() * 100,
            "Stuff PQ": np.nan_to_num(stuff_pq).mean() * 100,
            "Things PQ": np.nan_to_num(things_pq).mean() * 100,
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
    import dataset.pipelines.formatting

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
    data = KITTIDVPSDataset(
        pipeline=[
            dict(type='LoadMultiImagesDirect'),
            dict(type='LoadMultiAnnotationsDirect', with_depth=True, divisor=0),
            dict(type='SeqResizeWithDepth', img_scale=(1024, 2048), ratio_range=[1.0, 2.0], keep_ratio=True),
            dict(type='SeqFlipWithDepth', flip_ratio=0.5),
            dict(type='SeqRandomCropWithDepth', crop_size=(1024, 2048), share_params=True),
            dict(type='SeqNormalizeWithDepth', **img_norm_cfg),
            dict(type='SeqPadWithDepth', size_divisor=32),
            dict(
                type='VideoCollect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_depth', 'gt_instance_ids']),
            dict(type='ConcatVideoReferences'),
            dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
        ],
        data_root=os.path.expanduser('~/datasets/kitti-dvps'),
        split='val',
        ref_seq_index=[-1, 1]
    )
    np.set_string_function(lambda x: '<{} ; {}>'.format(x.shape, x.dtype))
    torch.set_printoptions(profile='short')
    for item in data:
        print(item)
