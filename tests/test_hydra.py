import os
import unittest

import hydra

import lmetk

CONFIG_DIR = os.path.join(".", "assets", "configs")


class TestBaseline(unittest.TestCase):
    _hydra_init = None

    @classmethod
    def setUpClass(cls):
        cls._hydra_init = hydra.initialize(version_base="1.3", config_path='../configs', job_name="run")
        cls._hydra_init.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._hydra_init.__exit__(None, None, None)

    def test_dm(self):
        data_name = 'cmapss_refactored'
        with self.subTest("cmapss"):

            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=[f'dataset={data_name}', f"paths=rt_local"], 
                return_hydra_config=True, 
            )

            print(f"Config file: {parameters.dataset.loader}")

            data_source = hydra.utils.instantiate(parameters.dataset.loader)
            data_source._load_data()
            data_source.X
            data_source.scaler
