import os.path as osp
import numpy as np

import mmcv
from mmdet.core import BitmapMasks

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    """Load multi images from file.
    Please refer to `mmdet.datasets.pipelines.loading.py:LoadImageFromFile`
    for detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in `results`, call the call function of
        `LoadImageFromFile` to load image.
        Args:
            results (list[dict]): List of dict from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains loaded image.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    """Sequence load annotations.
    Please refer to `mmdet.datasets.pipelines.loading.py:LoadAnnotations`
    for detailed docstring.
    Args:
        with_track (bool): If True, load instance ids of bboxes.
    """

    def __init__(self, with_track=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_track = with_track

    def _load_track(self, results):
        """Private function to load label annotations.
        Args:
            results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_instance_ids'] = results['ann_info']['instance_ids'].copy()

        return results

    def __call__(self, results):
        """Call function.
        For each dict in results, call the call function of `LoadAnnotations`
        to load annotation.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains loaded annotations, such as
            bounding boxes, labels, instance ids, masks and semantic
            segmentation annotations.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_track:
                _results = self._load_track(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class LoadRefImageFromFile(object):
    """
    Code reading reference frame information.
    Specific to Cityscapes-VPS, Cityscapes, and VIPER datasets.
    """

    def __init__(self, sample=True, to_float32=False):
        self.to_float32 = to_float32
        self.sample = sample

    def __call__(self, results):
        # requires dirname for ref images
        assert results['ref_prefix'] is not None, 'ref_prefix must be specified.'

        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = mmcv.imread(filename)
        # if specified by another ref json file.
        if 'ref_filename' in results['img_info']:
            ref_filename = osp.join(results['ref_prefix'],
                                    results['img_info']['ref_filename'])
            ref_img = mmcv.imread(ref_filename)  # [1024, 2048, 3]
        else:
            raise NotImplementedError('We need this implementation.')

        if self.to_float32:
            img = img.astype(np.float32)
            ref_img = ref_img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['ref_img'] = ref_img
        results['iid'] = results['img_info']['id']
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


def bitmasks2bboxes(bitmasks):
    bitmasks_array = bitmasks.masks
    boxes = np.zeros((bitmasks_array.shape[0], 4), dtype=np.float32)
    x_any = np.any(bitmasks_array, axis=1)
    y_any = np.any(bitmasks_array, axis=2)
    for idx in range(bitmasks_array.shape[0]):
        x = np.where(x_any[idx, :])[0]
        y = np.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = np.array((x[0], y[0], x[-1], y[-1]), dtype=np.float32)
    return boxes


@PIPELINES.register_module()
class LoadAnnotationsInstanceMasks:
    def __init__(self,
                 with_mask=True,
                 with_seg=True,
                 with_inst=False,
                 cherry=None,
                 file_client_args=dict(backend='disk')):
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.with_inst = with_inst
        self.file_client_args = file_client_args.copy()
        self.cherry = cherry
        self.file_client = None

    def _load_masks(self, results):
        """Private function to load mask annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        img_bytes = self.file_client.get(results['ann_info']['inst_map'])
        inst_mask = mmcv.imfrombytes(img_bytes, flag='unchanged').squeeze()
        if self.with_inst:
            results['gt_instance_map'] = inst_mask.copy().astype(int)
            results['gt_instance_map'][inst_mask < 10000] *= 1000
        if not self.with_mask:
            return results
        masks = []
        labels = []
        for inst_id in np.unique(inst_mask):
            if inst_id >= 10000:
                if self.cherry is not None and not (inst_id // 1000 in self.cherry):
                    continue
                masks.append((inst_mask == inst_id).astype(int))
                labels.append(inst_id // 1000)
        if len(masks) == 0:
            return None
        gt_masks = BitmapMasks(masks, height=inst_mask.shape[0], width=inst_mask.shape[1])
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        results['gt_labels'] = np.array(labels)

        boxes = bitmasks2bboxes(gt_masks)
        results['gt_bboxes'] = boxes
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        img_bytes = self.file_client.get(results['ann_info']['seg_map'])
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if self.with_mask or self.with_inst:
            results = self._load_masks(results)
            if results is None:
                return None
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        return repr_str
