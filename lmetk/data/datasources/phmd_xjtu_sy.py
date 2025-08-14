import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from lmetk.data.datasources.base.phmd_loader import PHMDMultiSourceLoader

logger = logging.getLogger(__name__)


class XJTU_SYLoader(PHMDMultiSourceLoader):
    """This class instantiates each of the soure"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _rename_columns(self, folds):
        return super()._rename_columns(folds)

    def _process_unit(self, df_unit, unit_name, features_col, target_col):
        return super()._process_unit(df_unit, unit_name, features_col, target_col)
