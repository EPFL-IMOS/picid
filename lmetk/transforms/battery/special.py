import logging
from collections import defaultdict
from typing import Any, Dict, Optional, override

from numpy import ndarray

from lmetk.data.data_objects.data import NamedTransformInput
from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.multisource import (
    MultiSourceTransformInterface,
    find_singular_ragged_dim,
)
from lmetk.utils.awkward_utils import ragged_index_tuples

# Set up a logger for better feedback
logger = logging.getLogger(__name__)


class UnflattenUnitsTransform(MultiSourceTransformInterface, BaseTransform):

    @override
    def fit_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @override
    def transform_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[ndarray]:

        out = [self.ragged2numpy(data) for data in data_segments]
        # Move from out=[[unti,unit,unit], [unti,unit]] to out=[[unti],[unit],[unit],[unti],[unit]]
        processed_out = []

        # 2. Loop through each dictionary in the original list
        for item in out:
            # 3. Get the list of keys from the current dictionary
            keys = item.keys()

            # 4. Unpack all the value lists and zip them together 💡
            #    zip(*[[d1, d2], [l1, l2], [id1, id2]]) becomes zip([d1, d2], [l1, l2], [id1, id2])
            for value_tuple in zip(*item.values()):

                # 5. Create a new dictionary by pairing the keys with the unzipped values
                #    dict(zip(['data', 'label', 'id'], (d1, l1, id1)))
                new_dict = dict(zip(keys, value_tuple))
                processed_out.append(new_dict)

        # Overwrite the original 'out' variable
        out = processed_out
        return out

    def ragged2numpy(self, data):
        # Ensure that all the keys have same dim 0
        assert len(set([len(val) for val in data.values()])) == 1, (
            f"All keys in data must have the same length at the first dimension. "
            f"Found lengths: {[len(val) for val in data.values()]}"
        )

        var_dim = find_singular_ragged_dim(data)
        multi_index = ragged_index_tuples(next(iter(data.values())), var_dim)

        out_segment = defaultdict(list)
        iterator = multi_index

        for coord in iterator:
            for k in data.keys():
                np_data = data[k][coord].to_numpy()
                out_segment[k].append(np_data)
        return out_segment

    @override
    def transform_data(
        self, data: NamedTransformInput, metadata: Dict[str, Any]
    ) -> Any:
        pass
