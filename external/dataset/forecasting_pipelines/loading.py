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
class LoadMultiImagesFromFile:
    """Load an image from file.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filenames = results['img_info']['filename']
        imgs = []
        for filename in filenames:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)
        img = np.concatenate(imgs, axis=-1)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'].append('img')
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotationsInstanceMasks:
    def __init__(self,
                 with_mask=True,
                 with_seg=True,
                 with_inst=False,
                 file_client_args=dict(backend='disk')):
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.with_inst = with_inst
        self.file_client_args = file_client_args.copy()
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
