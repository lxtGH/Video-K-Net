import cv2
import mmcv
import numpy as np
import warnings
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Normalize, Pad, RandomFlip, Resize


@PIPELINES.register_module()
class SeqColorAug(object):
    """Color augmention for images.
    Args:
        prob (list[float]): The probability to perform color augmention for
            each image. Defaults to [1.0, 1.0].
        rgb_var (list[list]]): The values of color augmentaion. Defaults to
            [[-0.55919361, 0.98062831, -0.41940627],
            [1.72091413, 0.19879334, -1.82968581],
            [4.64467907, 4.73710203, 4.88324118]].
    """

    def __init__(self,
                 prob=[1.0, 1.0],
                 rgb_var=[[-0.55919361, 0.98062831, -0.41940627],
                          [1.72091413, 0.19879334, -1.82968581],
                          [4.64467907, 4.73710203, 4.88324118]]):
        self.prob = prob
        self.rgb_var = np.array(rgb_var, dtype=np.float32)

    def __call__(self, results):
        """Call function.
        For each dict in results, perform color augmention for image in the
        dict.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains augmented color image.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']

            if self.prob[i] > np.random.random():
                offset = np.dot(self.rgb_var, np.random.randn(3, 1))
                # bgr to rgb
                offset = offset[::-1]
                offset = offset.reshape(3)
                image = (image - offset).astype(np.float32)

            _results['img'] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqBlurAug(object):
    """Blur augmention for images.
    Args:
        prob (list[float]): The probability to perform blur augmention for
            each image. Defaults to [0.0, 0.2].
    """

    def __init__(self, prob=[0.0, 0.2]):
        self.prob = prob

    def __call__(self, results):
        """Call function.
        For each dict in results, perform blur augmention for image in the
        dict.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains augmented blur image.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results['img']

            if self.prob[i] > np.random.random():
                sizes = np.arange(5, 46, 2)
                size = np.random.choice(sizes)
                kernel = np.zeros((size, size))
                c = int(size / 2)
                wx = np.random.random()
                kernel[:, c] += 1. / size * wx
                kernel[c, :] += 1. / size * (1 - wx)
                image = cv2.filter2D(image, -1, kernel)

            _results['img'] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqResize(Resize):
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
class SeqNormalize(Normalize):
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


@PIPELINES.register_module()
class SeqRandomFlip(RandomFlip):
    """Randomly flip for images.
    Please refer to `mmdet.datasets.pipelines.transfroms.py:RandomFlip` for
    detailed docstring.
    Args:
        share_params (bool): If True, share the flip parameters for all images.
            Defaults to True.
    """

    def __init__(self, share_params, *args, **kwargs):
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
class SeqPad(Pad):
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


@PIPELINES.register_module()
class SeqRandomCrop(object):
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
                 check_id_match=True
                 ):
        assert crop_size[0] > 0 and crop_size[1] > 0
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

        for key in results.get('img_fields', ['img']):
            img = results[key]
            if offsets is not None:
                offset_h, offset_w = offsets
            else:
                offset_h, offset_w = self.get_offsets(img)
            results['img_info']['crop_offsets'] = (offset_h, offset_w)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
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
        if self.share_params:
            offsets = self.get_offsets(results[0]['img'])
        else:
            offsets = None

        outs = []
        for _results in results:
            _results = self.random_crop(_results, offsets)
            if _results is None:
                return None
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
class SeqPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 share_params=True,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.share_params = share_params
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def get_params(self):
        """Generate parameters."""
        params = dict()
        # delta
        if np.random.randint(2):
            params['delta'] = np.random.uniform(-self.brightness_delta,
                                                self.brightness_delta)
        else:
            params['delta'] = None
        # mode
        mode = np.random.randint(2)
        params['contrast_first'] = True if mode == 1 else 0
        # alpha
        if np.random.randint(2):
            params['alpha'] = np.random.uniform(self.contrast_lower,
                                                self.contrast_upper)
        else:
            params['alpha'] = None
        # saturation
        if np.random.randint(2):
            params['saturation'] = np.random.uniform(self.saturation_lower,
                                                     self.saturation_upper)
        else:
            params['saturation'] = None
        # hue
        if np.random.randint(2):
            params['hue'] = np.random.uniform(-self.hue_delta, self.hue_delta)
        else:
            params['hue'] = None
        # swap
        if np.random.randint(2):
            params['permutation'] = np.random.permutation(3)
        else:
            params['permutation'] = None
        return params

    def photo_metric_distortion(self, results, params=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
            params (dict, optional): Pre-defined parameters. Default to None.
        Returns:
            dict: Result dict with images distorted.
        """
        if params is None:
            params = self.get_params()
        results['img_info']['color_jitter'] = params

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if params['delta'] is not None:
            img += params['delta']

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if params['contrast_first']:
            if params['alpha'] is not None:
                img *= params['alpha']

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if params['saturation'] is not None:
            img[..., 1] *= params['saturation']

        # random hue
        if params['hue'] is not None:
            img[..., 0] += params['hue']
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if not params['contrast_first']:
            if params['alpha'] is not None:
                img *= params['alpha']

        # randomly swap channels
        if params['permutation'] is not None:
            img = img[..., params['permutation']]

        results['img'] = img
        return results

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if self.share_params:
            params = self.get_params()
        else:
            params = None

        outs = []
        for _results in results:
            _results = self.photo_metric_distortion(_results, params)
            outs.append(_results)

        return outs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class ResizeWithRef(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        els = ['ref_img', 'img'] if 'ref_img' in results else ['img']
        for el in els:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[el], results['scale'], return_scale=True)
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[el], results['scale'], return_scale=True)
                scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                        dtype=np.float32)
            results[el] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        els = ['ref_bbox_fields', 'bbox_fields'] if 'ref_bbox_fields' in results else ['bbox_fields']
        for el in els:
            img_shape = results['img_shape']
            for key in results.get(el, []):
                bboxes = results[key] * results['scale_factor']
                bboxes[:, 0::2] = np.clip(
                    bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(
                    bboxes[:, 1::2], 0, img_shape[0] - 1)
                results[key] = bboxes

    def _resize_masks(self, results):
        els = ['ref_mask_fields', 'mask_fields'] if 'ref_mask_fields' in results else ['mask_fields']
        for el in els:
            for key in results.get(el, []):
                if results[key] is None:
                    continue
                if self.keep_ratio:
                    masks = [
                        mmcv.imrescale(
                            mask, results['scale_factor'],
                            interpolation='nearest')
                        for mask in results[key]
                    ]
                else:
                    mask_size = (results['img_shape'][1],
                                 results['img_shape'][0])
                    masks = [
                        mmcv.imresize(mask, mask_size,
                                      interpolation='nearest')
                        for mask in results[key]
                    ]
                results[key] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        # self._resize_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module()
class RandomFlipWithRef(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'])
            if 'ref_img' in results:
                results['ref_img'] = mmcv.imflip(results['ref_img'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            for key in results.get('ref_bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
            for key in results.get('ref_mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module()
class PadWithRef(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        els = ['ref_img', 'img'] if 'ref_img' in results else ['img']
        for el in els:
            if self.size is not None:
                padded_img = mmcv.impad(results['img'], self.size)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[el], self.size_divisor, pad_val=self.pad_val)
            results[el] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        els = ['ref_mask_fields', 'mask_fields'] if 'ref_mask_fields' in results else ['mask_fields']
        for el in els:
            pad_shape = results['pad_shape'][:2]
            for key in results.get(el, []):
                padded_masks = [
                    mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                    for mask in results[key]
                ]
                results[key] = np.stack(padded_masks, axis=0)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module()
class NormalizeWithRef(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(
            results['img'], self.mean, self.std, self.to_rgb)
        if 'ref_img' in results:
            results['ref_img'] = mmcv.imnormalize(
                results['ref_img'], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


@PIPELINES.register_module()
class RandomCropWithRef(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']

        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        ori_shape = img.shape
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        img_shape = img.shape
        results['img'] = img
        if 'ref_img' in results:
            ref_img = results['ref_img']
            ref_img = ref_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            results['ref_img'] = ref_img
        results['img_shape'] = img_shape
        results['crop_coords'] = [crop_y1, crop_y2, crop_x1, crop_x2]

        # crop bboxes accordingly and clip to the image boundary
        els = ['ref_bbox_fields', 'bbox_fields'] if 'ref_bbox_fields' in results else ['bbox_fields']
        for el in els:
            for key in results.get(el, []):
                bbox_offset = np.array(
                    [offset_w, offset_h, offset_w, offset_h],
                    dtype=np.float32)
                bboxes = results[key] - bbox_offset
                bboxes[:, 0::2] = np.clip(
                    bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(
                    bboxes[:, 1::2], 0, img_shape[0] - 1)
                results[key] = bboxes

        # filter out the gt bboxes that are completely cropped
        els = ['ref_bboxes', 'gt_bboxes'] if 'ref_bboxes' in results else ['gt_bboxes']
        for el in els:
            if el in results:
                gt_bboxes = results[el]
                valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                        gt_bboxes[:, 3] > gt_bboxes[:, 1])
                # if no gt bbox remains after cropping, just skip this image
                if not np.any(valid_inds):
                    return None
                results[el] = gt_bboxes[valid_inds, :]
                ell = el.replace('_bboxes', '_labels')
                if ell in results:
                    results[ell] = results[ell][valid_inds]
                #### filter gt_obj_ids just like gt_labes.
                elo = el.replace('_bboxes', '_obj_ids')
                if elo in results:
                    results[elo] = results[elo][valid_inds]
                # filter and crop the masks
                elm = el.replace('_bboxes', '_masks')
                if elm in results:
                    valid_gt_masks = []
                    for i in np.where(valid_inds)[0]:
                        gt_mask = results[elm][i][
                                  crop_y1:crop_y2, crop_x1:crop_x2]
                        valid_gt_masks.append(gt_mask)
                    results[elm] = valid_gt_masks

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module()
class PadFutureMMDet:
    """Pad the image & masks & segmentation map.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2], pad_val=pad_val)

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
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class KNetInsAdapter:
    """Adapter that is used to convert city-style instance class-ids
    to coco-style instance-ids (11-starting to 0-starting)
    """

    def __init__(self, stuff_nums=11):
        self.stuff_nums = stuff_nums

    def __call__(self, results):
        """Call function to modify gt_labels
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        results['gt_labels'] -= self.stuff_nums
        return results


@PIPELINES.register_module()
class KNetInsAdapterCherryPick:
    """Adapter that is used to convert city-style instance class-ids
    to coco-style instance-ids (11-starting to 0-starting)
    """

    def __init__(self, stuff_nums=11, cherry=(11, 13)):
        self.cherry = cherry
        self.stuff_nums = stuff_nums

    def __call__(self, results):
        """Call function to modify gt_labels
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        bias = 0
        for ch in self.cherry:
            results['gt_labels'][results['gt_labels'] == ch] -= bias
            bias += 1
        results['gt_labels'] -= self.stuff_nums
        return results
