from pprint import pprint

import numpy as np
import pytest
import torch

from torch.profiler import ProfilerActivity
from torch.profiler import profile as profile_f
from torch.profiler import record_function, schedule


from lmetk.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset


def test_init_dataset():
    data_dict = {
        "feature1": np.random.rand(100, 10),
        "feature2": np.random.rand(100, 5),
    }
    seq_len = 10
    label_len = 5
    pred_len = 5
    stride = 1

    ds = SlidingWindowBatchDataset(
        data_dict=data_dict,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        stride=stride,
    )

    assert False
    assert ds is not None
    assert len(ds) > 0


def test_dataloader_profile(tmp_path):
    # Create small dataset
    data_dict = {
        "feature1": np.random.rand(20000, 10),
        "feature2": np.random.rand(20000, 5),
    }
    ds = SlidingWindowBatchDataset(
        data_dict=data_dict,
        seq_len=10,
        label_len=5,
        pred_len=5,
        stride=1,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=0)
    pprint("len(dl): {}".format(len(dl)))

    prof = profile_f(
        activities=[ProfilerActivity.CPU],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )
    prof.start()

    for i, batch in enumerate(dl):
        with record_function("dataloader_iteration"):
            _ = batch
        prof.step()  # advance profiler state
        if i > 9:
            break

    prof.stop()
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # prof.export_chrome_trace(str(tmp_path / "trace.json"))
    prof.export_chrome_trace("trace.json")
    print("Profile finished")
    trace_uri = "file://" + str(tmp_path / "trace.json")
    print(f"Chrome trace file saved. Open in Finder: {trace_uri}")

    # Minimal assertion so test passes
    assert len(ds) > 0
