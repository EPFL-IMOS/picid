import os
import unittest

import hydra

from lmetk.data.datasources.multiunit_datasource import MultiUnitDataSource

CONFIG_DIR = os.path.join(".", "assets", "configs")


class TestBaseline(unittest.TestCase):
    _hydra_init = None

    @classmethod
    def setUpClass(cls):
        cls._hydra_init = hydra.initialize(
            version_base="1.3", config_path="../configs", job_name="run"
        )
        cls._hydra_init.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._hydra_init.__exit__(None, None, None)

    def test_dm(self):
        data_name = "cmapss_refactored"
        with self.subTest("cmapss"):
            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=[f"dataset={data_name}", f"paths=rt_local"],
                return_hydra_config=True,
            )

            print(f"Config file: {parameters.dataset.loader}")

            data_source = hydra.utils.instantiate(parameters.dataset.loader)
            data_source._load_data()
            data_source.X
            data_source.scaler

    def test_multi_unit_datasource(self):
        """
        Combine two identical CMAPSSDataSource instances and verify combined behavior.
        """
        data_name = "cmapss_refactored"
        parameters = hydra.compose(
            config_name="run.yaml",
            overrides=[f"dataset={data_name}", "paths=rt_local"],
            return_hydra_config=True,
        )
        # Instantiate two base sources
        src1 = hydra.utils.instantiate(parameters.dataset.loader)
        src2 = hydra.utils.instantiate(parameters.dataset.loader)
        # Prepare loader arguments, splitter, and scaler from one source
        load_args = parameters.dataset.loader
        splitter = src1.data_splitter
        scaler = src1.scaler

        # Create MultiUnitDataSource
        multi_src = MultiUnitDataSource(
            sources=[src1, src2],
            load_arguments=load_args,
            data_splitter=splitter,
            scaler=scaler,
        )
        # Run pipeline and retrieve splits
        data_dict, timestamps_dict = multi_src.get_data()

        # Expect train/val/test keys
        for split in ("train", "val", "test"):
            self.assertIn(split, data_dict)
            self.assertIn(split, timestamps_dict)
            # Combined length should be roughly sum of individual sources
            len_single = len(src1._load_data()[0])
            self.assertEqual(
                len(data_dict[split]),
                len_single
                * 2
                // 3,  # approximate: train/val/test split ratios applied twice
            )

        # Scaler should be fitted
        self.assertIsNotNone(multi_src.get_scaler())
