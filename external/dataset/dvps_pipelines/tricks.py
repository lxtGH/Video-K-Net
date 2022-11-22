import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import AutoAugment


@PIPELINES.register_module()
class SeqAutoAug(AutoAugment):
    """
    Auto augmentation a sequence.
    """
    def __init__(self, policies):
        super().__init__(policies=policies)

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        outs = []
        for _results in results:
            out = transform(_results)
            outs.append(out)
        return outs
