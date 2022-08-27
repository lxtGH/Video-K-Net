import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize, RandomFlip, Pad, Normalize


@PIPELINES.register_module()
class ResizeWithDepth(Resize):
    """This subclass of Resize is to support depth resize
    """

    def __init__(self, *args, **kwargs):
        assert kwargs['keep_ratio']
        super().__init__(*args, **kwargs)

    def _resize_depth(self, results):
        """Resize depth with ``results['scale']``"""
        # Although depth is not discrete, we use nearest to match the segmentation
        for key in results.get('depth_fields', []):
            if self.keep_ratio:
                results[key] = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                results[key] = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results[key] /= results['scale_factor'].mean()

    def __call__(self, results):
        super().__call__(results)
        self._resize_depth(results)
        return results


@PIPELINES.register_module()
class SeqResizeWithDepth(ResizeWithDepth):
    """Resize images.
    Please refer to `mmdet.datasets.pipelines.transfroms.py:Resize` for
    detailed docstring.
    Args:
        share_params (bool): If True, share the resize parameters for all
            images. Defaults to True.
    """

    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.
        For each dict in results, call the call function of `Resize` to resize
        image and corresponding annotations.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains resized results,
            'img_shape', 'pad_shape', 'scale_factor', 'keep_ratio' keys
            are added into result dict.
        """
        outs, scale = [], None
        for i, _results in enumerate(results):
            if self.share_params and i > 0:
                _results['scale'] = scale
            _results = super().__call__(_results)
            if self.share_params and i == 0:
                scale = _results['scale']
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class RandomFlipWithDepth(RandomFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        super().__call__(results)
        if results['flip']:
            for key in results.get('depth_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results


@PIPELINES.register_module()
class SeqFlipWithDepth(RandomFlipWithDepth):
    """Randomly flip for images.
    Please refer to `mmdet.datasets.pipelines.transfroms.py:RandomFlip` for
    detailed docstring.
    Args:
        share_params (bool): If True, share the flip parameters for all images.
            Defaults to True.
    """

    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.
        For each dict in results, call `RandomFlip` to randomly flip image.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains flipped results, 'flip',
            'flip_direction' keys are added into the dict.
        """
        if self.share_params:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            flip = cur_dir is not None
            flip_direction = cur_dir

            for _results in results:
                _results['flip'] = flip
                _results['flip_direction'] = flip_direction

        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomCropWithDepth(object):
    """Sequentially random crop the images & bboxes & masks.
    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.
    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        share_params (bool, optional): Whether share the cropping parameters
            for the images.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 allow_negative_crop=False,
                 share_params=False,
                 bbox_clip_border=True,
                 check_id_match=True,
                 ):
        assert crop_size is None or (crop_size[0] > 0 and crop_size[1] > 0)
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop
        self.share_params = share_params
        self.bbox_clip_border = bbox_clip_border
        self.check_id_match = check_id_match
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': ['gt_labels', 'gt_instance_ids'],
            'gt_bboxes_ignore': ['gt_labels_ignore', 'gt_instance_ids_ignore']
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def get_offsets(self, img):
        """Random generate the offsets for cropping."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_w

    def random_crop(self, results, offsets=None):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
            offsets (tuple, optional): Pre-defined offsets for cropping.
                Default to None.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """
        # Only supporting img
        assert results['img_fields'] == ['img']
        img = results['img']
        if offsets is not None:
            offset_h, offset_w = offsets
        else:
            offset_h, offset_w = self.get_offsets(img)
        results['crop_offsets'] = (offset_h, offset_w)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                    bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # self.allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not self.allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_keys = self.bbox2label.get(key)
            for label_key in label_keys:
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # crop depth
        for key in results.get('depth_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def __call__(self, results):
        """Call function to sequentially randomly crop images, bounding boxes,
        masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """
        if self.share_params and self.crop_size is not None:
            offsets = self.get_offsets(results[0]['img'])
        else:
            offsets = None

        if self.crop_size is not None:
            outs = []
            for _results in results:
                _results = self.random_crop(_results, offsets)
                if _results is None:
                    return None
                outs.append(_results)
        else:
            outs = []
            for _results in results:
                outs.append(_results)

        if len(outs) == 2 and self.check_id_match:
            ref_result, result = outs[1], outs[0]
            if self.check_match(ref_result, result):
                return None

        return outs

    def check_match(self, ref_results, results):
        ref_ids = ref_results['gt_instance_ids'].tolist()
        gt_ids = results['gt_instance_ids'].tolist()
        gt_pids = [ref_ids.index(i) if i in ref_ids else -1 for i in gt_ids]
        nomatch = (np.array(gt_pids) == -1).all()
        return nomatch


@PIPELINES.register_module()
class PadWithDepth(Pad):

    def _pad_depth(self, results):
        """Pad depth according to
        ``results['pad_shape']``."""
        for key in results.get('depth_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2], pad_val=0)

    # the original pad sem_seg does not consider the no_obj_class with value except for 0
    #
    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        no_obj_class = results['no_obj_class']
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=no_obj_class)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        self._pad_depth(results)
        return results


@PIPELINES.register_module()
class SeqPadWithDepth(PadWithDepth):
    """Pad images.
    Please refer to `mmdet.datasets.pipelines.transfroms.py:Pad` for detailed
    docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in results, call the call function of `Pad` to pad image.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains padding results,
            'pad_shape', 'pad_fixed_size' and 'pad_size_divisor' keys are
            added into the dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


# There is nothing new from SeqNormalize.
@PIPELINES.register_module()
class SeqNormalizeWithDepth(Normalize):
    """Normalize images.
    Please refer to `mmdet.datasets.pipelines.transfroms.py:Normalize` for
    detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in results, call the call function of `Normalize` to
        normalize image.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains normalized results,
            'img_norm_cfg' key is added into result dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs
