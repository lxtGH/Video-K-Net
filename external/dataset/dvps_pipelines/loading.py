import mmcv
import numpy as np
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES


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
class LoadImgDirect:
    """Go ahead and just load image
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict requires "img" which is the img path.

        Returns:
            dict: The dict contains loaded image and meta information.
            'img' : img
            'img_shape' : img_shape
            'ori_shape' : original shape
            'img_fields' : the img fields
        """
        img = mmcv.imread(results['img'], channel_order='rgb', flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', ")
        return repr_str


@PIPELINES.register_module()
class LoadMultiImagesDirect(LoadImgDirect):
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
class LoadAnnotationsDirect:
    """Go ahead and just load image
    """

    def __init__(self,
                 with_depth=True,
                 divisor: int = 1000,
                 cherry_pick=False,
                 cherry=None,
                 viper=False,
                 vipseg=False
                 ):
        self.with_depth = with_depth
        self.panseg_divisor = divisor
        self.cherry_pick = cherry_pick
        self.cherry = cherry
        self.viper = viper
        self.vipseg=vipseg
        if self.vipseg:
            self.panseg_divisor = 1000

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict requires "img" which is the img path.

        Returns:
            dict: The dict contains loaded image and meta information.
            'depth_fields' : the depth fields for supporting depth aug
        """

        if self.with_depth:
            depth = mmcv.imread(results['depth'], flag='unchanged').astype(np.float32) / 256.
            del results['depth']
            depth[depth >= 80.] = 80.
            results['gt_depth'] = depth
            results['depth_fields'] = ['gt_depth']

        local_divisor = 10000
        if self.panseg_divisor == 0:
            # The seperate file to store class id and inst id
            gt_semantic_seg = mmcv.imread(results['ann_class'], flag='unchanged').astype(np.float32)
            inst_map = mmcv.imread(results['ann_inst'], flag='unchanged').astype(np.float32)
            ps_id = gt_semantic_seg * local_divisor + inst_map
            del results['ann_class']
            del results['ann_inst']
        elif self.panseg_divisor == -1:
            # KITTI step mode which means the panseg is stored with RGB
            id_map = mmcv.imread(results['ann'], flag='color', channel_order='rgb')
            gt_semantic_seg = id_map[..., 0].astype(np.float32)
            inst_map = id_map[..., 1].astype(np.float32) * 256 + id_map[..., 2].astype(np.float32)
            ps_id = gt_semantic_seg * local_divisor + inst_map
            del results['ann']
        else:
            ps_id = mmcv.imread(results['ann'], flag='unchanged').astype(np.float32)
            if self.vipseg:
                ps_id = results['pre_hook'](ps_id)
                del results['pre_hook']
            # This is for viper
            if self.viper or self.vipseg:
                ps_id[ps_id < 1000] *= 1000
            del results['ann']
            gt_semantic_seg = ps_id // self.panseg_divisor

        if self.viper:
            gt_semantic_seg[gt_semantic_seg >= results['thing_upper']] = results['no_obj_class']
        results['gt_semantic_seg'] = gt_semantic_seg.astype(np.int)
        results['seg_fields'] = ['gt_semantic_seg']

        classes = []
        masks = []
        instance_ids = []
        no_obj_class = results['no_obj_class']
        for pan_seg_id in np.unique(ps_id):
            classes.append(pan_seg_id // self.panseg_divisor if self.panseg_divisor > 0
                           else pan_seg_id // local_divisor)
            masks.append((ps_id == pan_seg_id).astype(np.int))
            instance_ids.append(pan_seg_id)
        gt_labels = np.stack(classes).astype(np.int)
        gt_instance_ids = np.stack(instance_ids).astype(np.int)
        gt_masks = BitmapMasks(masks, height=results['img_shape'][0], width=results['img_shape'][1])
        # check the sanity of gt_masks
        verify = np.sum(gt_masks.masks.astype(np.int), axis=0)
        assert (verify == np.ones(gt_masks.masks.shape[-2:], dtype=verify.dtype)).all()
        # now delete the no_obj_class
        gt_masks.masks = np.delete(gt_masks.masks, gt_labels == no_obj_class, axis=0)
        gt_instance_ids = np.delete(gt_instance_ids, gt_labels == no_obj_class)
        gt_labels = np.delete(gt_labels, gt_labels == no_obj_class)
        if results['is_instance_only'] and not self.cherry_pick:
            gt_masks.masks = np.delete(
                gt_masks.masks,
                (gt_labels >= results['thing_upper']) | (gt_labels < results['thing_lower']),
                axis=0
            )
            gt_instance_ids = np.delete(
                gt_instance_ids,
                (gt_labels >= results['thing_upper']) | (gt_labels < results['thing_lower'])
            )
            gt_labels = np.delete(
                gt_labels,
                (gt_labels >= results['thing_upper']) | (gt_labels < results['thing_lower'])
            )
            gt_labels -= results['thing_lower']
        elif results['is_instance_only'] and self.cherry_pick:
            gt_masks.masks = np.delete(
                gt_masks.masks,
                list(map(lambda x: x not in self.cherry, gt_labels)),
                axis=0
            )
            gt_instance_ids = np.delete(
                gt_instance_ids,
                list(map(lambda x: x not in self.cherry, gt_labels)),
            )
            gt_labels = np.delete(
                gt_labels,
                list(map(lambda x: x not in self.cherry, gt_labels)),
            )
            gt_labels = np.array(list(map(lambda x: self.cherry.index(x), gt_labels))) if len(gt_labels) > 0 else []

        if len(gt_labels) == 0:
            return None

        results['gt_labels'] = gt_labels
        results['gt_masks'] = gt_masks
        results['gt_instance_ids'] = gt_instance_ids
        results['mask_fields'] = ['gt_masks']

        # generate boxes
        boxes = bitmasks2bboxes(gt_masks)
        results['gt_bboxes'] = boxes
        results['bbox_fields'] = ['gt_bboxes']
        return results


@PIPELINES.register_module()
class LoadMultiAnnotationsDirect(LoadAnnotationsDirect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if _results is None:
                return None
            outs.append(_results)
        return outs
