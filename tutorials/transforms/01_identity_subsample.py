#!/usr/bin/env python3
"""Tutorial: Apply IdentityPassThrough and SubsampleTransform."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base_transforms.identity import IdentityPassThrough
from picid.transforms.base_transforms.subsample import SubsampleTransform


def main():
    data = NamedTransformInput(
        features=np.arange(120).reshape(120, 1).astype(np.float32)
    )
    identity = IdentityPassThrough()
    out1 = identity.transform_data(data, {"mode": "train"})
    assert out1["features"].shape == data["features"].shape
    subsample = SubsampleTransform(step=4)
    out2 = subsample.transform_data(data.copy(), {"mode": "train"})
    assert len(out2["features"]) == 30
    print("OK")


if __name__ == "__main__":
    main()
