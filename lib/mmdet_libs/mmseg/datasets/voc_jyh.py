import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalVOCDataset_jyh(CustomDataset):  # +++++++++++++++++++++++
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'circle_max', 'circle_min')  # -------------------------------

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]  # ------------------------------------

    def __init__(self, split, **kwargs):
        super(PascalVOCDataset_jyh, self).__init__(     # +++++++++++++++++++++++++++++
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
