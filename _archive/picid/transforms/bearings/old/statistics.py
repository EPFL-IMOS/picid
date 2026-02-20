import logging
import numpy as np
import pandas as pd

from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logger = logging.getLogger(__name__)


class StatisticsBearingsTransform(DenseTransform, NoFitPerSegmentMixin):
    """
    A pass-through transform that prints statistics about the data.

    This transform does not modify the data; it inspects the data and
    prints key statistics like shape, dtype, mean, std, etc., to the console
    before passing the original data to the next step in the pipeline.
    """

    def __init__(self, name: str, enabled: bool = True, **kwargs):
        """
        Initialize the transform.

        Args:
            name (str): A descriptive name for this instance of the transform,
                        used in the print output to identify the pipeline step.
            enabled (bool): If False, the transform will do nothing and just
                            pass the data through. Defaults to True.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)
        self.name = name
        self.enabled = enabled

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        """
        Prints statistics and returns the original, unmodified data.

        Args:
            data: Dictionary containing a single numpy array or pandas object.
            metadata: Dictionary containing auxiliary metadata (unused).

        Returns:
            The original, unmodified numpy array or pandas object.
        """
        # If the transform is disabled, just pass the data through immediately.
        if not self.enabled:
            # Assuming data has one entry, as in the ReshapeTransform example.
            return next(iter(data.values()))

        # Ensure there's only one item to process, matching the base transform's logic.
        assert (
            len(data) == 1
        ), "StatisticsTransform expects data with exactly one entry."

        key, value = next(iter(data.items()))

        print("\n" + "=" * 80)
        print(f"📊 STATISTICS FROM: '{self.name}' (Input key: '{key}')")
        print("-" * 80)

        # Basic Information
        print(f"  - Type: {type(value)}")
        if hasattr(value, "shape"):
            print(f"  - Shape: {value.shape}")
        if hasattr(value, "dtype"):
            print(f"  - Dtype: {value.dtype}")
        if hasattr(value, "size"):
            print(f"  - Total Elements: {value.size}")

        # Check if the data is numeric before calculating stats
        is_numeric = False
        if hasattr(value, "dtype"):
            is_numeric = np.issubdtype(value.dtype, np.number)

        if is_numeric:
            print("\n  Numeric Statistics:")
            # Use np.nan* functions for robustness against missing values
            try:
                print(f"    - Mean:   {np.nanmean(value):.4f}")
                print(f"    - Std Dev:  {np.nanstd(value):.4f}")
                print(f"    - Min:      {np.nanmin(value):.4f}")
                print(f"    - 25%:      {np.nanpercentile(value, 25):.4f}")
                print(f"    - Median:   {np.nanmedian(value):.4f}")
                print(f"    - 75%:      {np.nanpercentile(value, 75):.4f}")
                print(f"    - Max:      {np.nanmax(value):.4f}")
                print(f"    - Sum:      {np.nansum(value):.4f}")
            except TypeError:
                print(
                    "    - Could not compute numeric stats (likely object array with mixed types)."
                )
        else:
            print("\n  - Data is non-numeric, skipping numeric statistics.")
            # For pandas objects, we can show other useful info
            if isinstance(value, (pd.DataFrame, pd.Series)):
                if not value.empty:
                    print(f"  - Unique values: {value.nunique()}")
                    print("  - Value Counts (top 5):")
                    print(
                        value.value_counts().head(5).to_string().replace("\n", "\n    ")
                    )

        print("=" * 80 + "\n")

        # IMPORTANT: Return the original, unmodified value
        return value

    def __call__(self, data: NamedTransformInput, metadata: dict) -> np.ndarray:
        return self.transform_data(data, metadata)
