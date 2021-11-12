import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalVOCDataset_daoxian(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'xian_luo', 'xian_bao')  # --------------------------------------------------------

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]  # ------------------------------------------------------------

    def __init__(self, split, **kwargs):
        super(PascalVOCDataset_daoxian, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
