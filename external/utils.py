import io

from panopticapi.utils import id2rgb
from PIL import Image


def encode_panoptic(panoptic_results):
    panoptic_img, segments_info = panoptic_results
    with io.BytesIO() as out:
        Image.fromarray(id2rgb(panoptic_img)).save(out, format='PNG')
        return out.getvalue(), segments_info
