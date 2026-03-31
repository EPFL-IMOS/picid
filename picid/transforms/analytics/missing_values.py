import logging
import os
import numpy as np
import awkward as ak
from typing import Dict, List, Optional, Any
from collections import defaultdict

from picid.data.data_objects import NamedTransformInput
from picid.data.data_objects.utils import convert_to_numpy
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

logger = logging.getLogger(__name__)


class MissingValuesStatsLogger(NoFitPerSegmentMixin, DenseTransform):
    """
    Log missing-value statistics without modifying the input data.

    Parameters
    ----------
    saving_path : str
        Root directory where the report will be written.
    apply_to : list[str], optional
        Optional subset of keys to inspect.
    filename : str, default="missing_values_stats.txt"
        Name of the output report file.
    **kwargs
        Additional keyword arguments forwarded to the base transform.

    Notes
    -----
    This transform is a pass-through inspector. It does not modify the data.
    It records both per-unit statistics, where the NaN count is related to the
    individual unit length, and global statistics aggregated across the full
    dataset. Results are printed to the console and saved to a text report
    under the configured logs directory.
    """

    def __init__(
        self,
        saving_path: str,
        apply_to: Optional[List[str]] = None,
        filename: str = "missing_values_stats.txt",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.saving_path = saving_path
        self.apply_to = apply_to
        self.filename = filename

        # Ensure log directory exists
        self.log_dir = os.path.join(self.saving_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def transform_data(
        self, data: NamedTransformInput, metadata: Dict
    ) -> NamedTransformInput:
        """
        Return the input unchanged for single-source execution.

        Parameters
        ----------
        data : NamedTransformInput
            Input mapping to pass through unchanged.
        metadata : dict
            Metadata dictionary, preserved for interface compatibility.

        Returns
        -------
        NamedTransformInput
            The input mapping, unchanged.
        """
        return data

    def transform_multi_source(
        self,
        data_segments: list[NamedTransformInput],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[list[NamedTransformInput], dict[str, Any]]:
        """
        Compute and persist missing-value statistics for each segment.

        Parameters
        ----------
        data_segments : list[NamedTransformInput]
            Segment list to inspect.
        metadata : dict, optional
            Optional pipeline metadata.

        Returns
        -------
        tuple[list[NamedTransformInput], dict[str, Any]]
            The original segments and an empty metadata update.
        """

        # 1. Setup Logging Buffer
        output_lines = []
        separator = "=" * 105
        # slightly wider columns for shape/type info
        header = f"{'Unit ID':<20} | {'Key':<15} | {'Shape/Type':<25} | {'NaN Count':<12} | {'% Missing':<10}"

        output_lines.append(separator)
        output_lines.append("MISSING VALUES STATISTICS REPORT")
        output_lines.append(separator)
        output_lines.append(header)
        output_lines.append("-" * 105)

        # 2. Global Accumulators
        global_stats = defaultdict(lambda: {"total_nans": 0, "total_size": 0})

        # 3. Iterate over units (data segments)
        for i, segment in enumerate(data_segments):
            # Try to find a Unit ID in metadata, fallback to index
            seg_metadata = segment.metadata if segment.metadata else {}
            unit_id = str(seg_metadata.get("unit_id", f"Unit_{i}"))

            # Determine keys to process
            keys_to_process = (
                self.apply_to if self.apply_to is not None else segment.keys()
            )

            for key in keys_to_process:
                if key not in segment:
                    continue

                val = segment[key]

                try:
                    n_nans = 0
                    total_size = 0
                    shape_str = "?"

                    # --- CASE A: Awkward Array (Ragged) ---
                    if isinstance(val, ak.Array):
                        # Flatten completely to count all elements regardless of nesting depth
                        val_flat = ak.flatten(val, axis=None)
                        total_size = len(val_flat)

                        # Count NaNs (robust to different numeric types)
                        # ak.isnan returns boolean array, ak.sum counts True
                        if total_size > 0:
                            # Ensure we are checking a numeric type or float
                            n_nans = ak.sum(np.isnan(val_flat))

                        # Use the Awkward type as the "Shape" description
                        # Truncate if too long (e.g. 100 * var * float64)
                        shape_str = str(val.type)
                        if len(shape_str) > 25:
                            shape_str = "Ragged (Awkward)"

                    # --- CASE B: NumPy Array (Dense) ---
                    else:
                        # Convert safely if it's a list or similar, but avoid forcing 2D if not needed
                        np_val = convert_to_numpy(val, ensure_2d=False)
                        total_size = np_val.size
                        n_nans = np.isnan(np_val).sum()
                        shape_str = str(np_val.shape)

                    # Calculate Percentage
                    pct = (n_nans / total_size * 100) if total_size > 0 else 0.0

                    # Log Per-Unit row
                    line = f"{unit_id:<20} | {key:<15} | {shape_str:<25} | {n_nans:<12} | {pct:>9.2f}%"
                    output_lines.append(line)

                    # Update Global Stats
                    global_stats[key]["total_nans"] += int(n_nans)
                    global_stats[key]["total_size"] += int(total_size)

                except Exception as e:
                    # Catch-all to prevent one bad unit crashing the whole pipeline
                    # We log the specific error but continue processing
                    logger.warning(f"Stats Error [Unit: {unit_id}, Key: {key}]: {e}")
                    output_lines.append(
                        f"{unit_id:<20} | {key:<15} | {'ERROR':<25} | {'-':<12} | {'-':<10}"
                    )

        # 4. Compile Global Statistics
        output_lines.append(separator)
        output_lines.append("GLOBAL SUMMARY")
        output_lines.append(separator)
        summary_header = f"{'Key':<20} | {'Total NaNs':<15} | {'Total Points':<15} | {'Global % Missing':<15}"
        output_lines.append(summary_header)
        output_lines.append("-" * 80)

        for key, stats in global_stats.items():
            total_nans = stats["total_nans"]
            total_size = stats["total_size"]
            global_pct = (total_nans / total_size * 100) if total_size > 0 else 0.0

            summary_line = f"{key:<20} | {total_nans:<15} | {total_size:<15} | {global_pct:>13.2f}%"
            output_lines.append(summary_line)

        output_lines.append(separator)

        # 5. Output to Console and File
        final_output = "\n".join(output_lines)

        # Print nicely to console
        print(final_output)

        # Save to file
        file_path = os.path.join(self.log_dir, self.filename)
        try:
            with open(file_path, "w") as f:
                f.write(final_output)
            logger.info(f"Missing values statistics saved to: {file_path}")
        except IOError as e:
            logger.error(f"Failed to write statistics to file: {e}")

        # 6. Return data unmodified (Pass-through)
        return data_segments, {}
