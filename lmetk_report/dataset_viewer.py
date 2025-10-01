from io import StringIO
import os
from pathlib import Path
import logging

import holoviews as hv
from omegaconf import OmegaConf
import panel as pn
import param
from hydra import compose, initialize
from hydra.utils import instantiate
import pandas as pd
import numpy as np
import awkward as ak
import hvplot.pandas

from rich.console import Console

from lmetk.data.data_objects.data import DatasetContainer
from lmetk.data.preprocessing.preprocessor import PreProcessor
from lmetk.transforms.base.transform_manager import ConfigTransformManager
from lmetk.utils.awkward_utils import ak_find_var_dims
from lmetk.utils.rich_output import print_data_dict_structure
import pickle

FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@pn.cache
def reconfig_basic_config(format_=FORMAT, level=logging.INFO):
    """(Re-)configure logging"""
    logging.basicConfig(format=format_, level=level, force=True)
    logging.info("Logging.basicConfig completed successfully")


reconfig_basic_config()
logger = logging.getLogger(name="app")

# Initialize Panel and HoloViews
pn.extension()
hv.extension("bokeh")

print(f"Current working directory: {os.getcwd()}")
os.environ["PROJECT_ROOT"] = Path(".").absolute().as_posix()
print(f"PROJECT_ROOT={os.environ['PROJECT_ROOT']}")


def load_dataset_from_cfg(cfg: OmegaConf) -> dict:
    preprocessor_mode = "per_unit"
    dataset_source = instantiate(cfg.datasource)

    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    preprocessor = PreProcessor(preprocessor_mode, dataset_source, transforms_manager)

    preprocessor = PreProcessor(preprocessor_mode, dataset_source, transforms_manager)

    # Get cache dirs and flags
    data_cache_path = (
        cfg.paths.cache_path
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    data_library_part_path = (
        Path(cfg.paths.root_dir) / "lmetk/data/datasources"
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    transform_library_part_path = (
        Path(cfg.paths.root_dir) / "lmetk/transforms"
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )

    logger.info(f"Data cache path: {data_cache_path}")
    logger.info(f"Data library path: {data_library_part_path}")
    logger.info(f"Transform library path: {transform_library_part_path}")

    cache_preprocessed_flag = cfg.cache.use_cache_after_transfroms

    # Run the preprocessing pipeline
    preprocessor.pipeline(
        data_cache_path=data_cache_path,
        data_library_part_path=data_library_part_path,
        transform_library_part_path=transform_library_part_path,
        cache_preprocessed=cache_preprocessed_flag,
    )

    # TODO: return_splits_on_first_level is not clea..
    data_dict = preprocessor.get_processed_data_dict(return_splits_on_first_level=True)
    meta_data_dict = preprocessor.get_meta_data_dict()

    return data_dict


pn.extension("hvplot")


class TimeSeriesViewer(pn.viewable.Viewer):

    train_val_test_selector = param.Selector(
        objects=["train", "val", "test"], default="train"
    )

    data = param.Dict(default={})
    split_data: ak.Array = param.Parameter()

    segment_slider = param.Parameter()
    zoom = param.Parameter()
    unit_slider = param.Parameter()

    def __init__(self, **params):
        super().__init__(**params)

        # Persistent plot pane
        self.plot_pane = pn.pane.HoloViews(hv.Curve([]), sizing_mode="stretch_both")

        self.param.watch(self._data_updated, "data")
        self.param.watch(self._unit_data_updated, "split_data")

    def _data_updated(self, event):
        data = event.obj.data
        if not data:
            return
        print("Data updated, updating widgets...")
        self.unit_slider = pn.widgets.IntSlider(
            name="Unit", start=0, end=len(data) - 1, value=0
        )
        self.split_data = data.get(self.train_val_test_selector, {})

    def _unit_data_updated(self, event):
        if not self.split_data:
            return
        data = self.split_data["features"][self.unit_slider.value]
        self.segment_slider = pn.widgets.IntSlider(
            name="Segment", start=0, end=len(data) - 1, value=0
        )
        self.zoom = pn.widgets.RangeSlider(
            name="Zoom (x)",
            start=0,
            end=len(data) - 1,
            value=(0, len(data) - 1),
        )

        # Watch slider updates
        self.unit_slider.param.watch(self._update_plot_slider, "value")
        self.segment_slider.param.watch(self._update_plot_slider, "value")
        self.zoom.param.watch(self._update_plot_zoom, "value")

        # Initial plot update
        self.plot_pane.object = self._get_plot(xlim=self.zoom.value)

    def _get_plot(self, unit=None, segment=None, xlim=None):
        if self.data is None or not self.split_data:
            return hv.Curve([])

        if unit is None:
            unit = self.unit_slider.value
        if segment is None:
            segment = self.segment_slider.value

        plot_data = self.split_data["features"][unit][segment]
        df = pd.DataFrame(ak.to_numpy(plot_data))
        plot = df.hvplot(width=600, height=300)
        if xlim is not None:
            plot = plot.opts(xlim=xlim)
        return plot

    def _update_plot_slider(self, event):
        segment = self.segment_slider.value
        unit = self.unit_slider.value
        plot_data = self.split_data["features"][unit][segment]

        self.zoom.start = 0
        self.zoom.end = len(plot_data) - 1
        self.zoom.value = (0, len(plot_data) - 1)

        self.plot_pane.object = self._get_plot(xlim=self.zoom.value)

    def _update_plot_zoom(self, event):
        self.plot_pane.object = self._get_plot(xlim=event.new)

    @param.depends("split_data", watch=True)
    def controls_panel(self):
        return pn.Row(
            pn.Param(self, parameters=["train_val_test_selector"]),
            self.unit_slider if self.unit_slider else pn.pane.Markdown("_No data yet_"),
            (
                self.segment_slider
                if self.segment_slider
                else pn.pane.Markdown("_No data yet_")
            ),
            self.zoom if self.zoom else pn.pane.Markdown("_No data yet_"),
        )

    def __panel__(self):
        return pn.Column(
            self.controls_panel,
            pn.Row(self.plot_pane),
        )


# Class-based Approach Using `param.Parameterized`
class DatasetViewer(pn.viewable.Viewer):
    dataset_selector = param.Selector(objects=[], default=None)
    load_dataset_button = param.Action(
        lambda x: x.param.trigger("load_dataset_button"), label="Load Dataset"
    )

    # @param.depends("load_dataset_button", watch=True)
    # def on_load_dataset_button_pressed(self):
    #     """Load dataset when the button is pressed."""
    #     self.timeseries_viewer.dataset = self.load_dataset(self.dataset_selector)

    def __init__(self, **params):
        super().__init__(**params)

        experiment_aliases = {
            "nb14/reconstruction/residual_autoencoder": "reconstruction_nb14",
            "pronostia/reconstruction/residual_autoencoder": "reconstruction_nb14",
            "xjtu_sy/reconstruction/residual_autoencoder": "reconstruction_nb14",
            "unibo/reconstruction/residual_autoencoder": "reconstruction_nb14",
        }
        # update the parameter objects (not overwrite)
        self.param.dataset_selector.objects = list(experiment_aliases.keys())
        self.dataset_selector = list(experiment_aliases.keys())[0]  # set default value
        self.data_structure_description = pn.Column("No data loaded yet")
        self.data_dict = {}
        self.timeseries_viewer = TimeSeriesViewer()

    def plot_graph(self, dataset: any):
        return hv.Curve([])  # Placeholder for actual plot

    def load_config(self, exp_alias: str):

        def load_cfg(overrides):
            with initialize(version_base="1.2", config_path="../configs"):
                cfg = compose(config_name="run.yaml", overrides=overrides)
                return cfg

        cfg = load_cfg([f"experiment={exp_alias}", "paths=rt_local", "debug=nb14"])

        logger.info(f"Loading cfg with alias: {exp_alias}")
        return cfg

    @param.depends("load_dataset_button", watch=True)
    def _on_load_dataset_button_pressed(self):
        # self.data_dict = load_dataset_from_cfg(self.cfg)
        print("Loading dataset...")
        with open("testing_data.pkl", "rb") as f:
            self.data_dict = pickle.load(f)

        self.timeseries_viewer.data = self.data_dict
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None)
        tree = print_data_dict_structure(self.data_dict)
        console.print(tree)
        self.data_structure_description.clear()
        self.data_structure_description.append(pn.pane.Str(buf.getvalue(), width=400))

    @param.depends("dataset_selector")
    def config_pane(self):
        """Display the configuration of the selected dataset."""
        cfg = self.load_config(self.dataset_selector)
        self.cfg = cfg  # Store cfg for later use
        cfg_json = OmegaConf.to_container(cfg.datasource, resolve=True)
        json_pane = pn.pane.JSON(cfg_json, name="Configuration", depth=5, width=400)
        logger.info(f"Loaded config for dataset: {self.dataset_selector}")
        return json_pane

    def __panel__(self):
        """Panel layout structure."""
        return pn.Row(
            pn.Column(
                pn.Param(self.param),
                self.config_pane,
                self.data_structure_description,
            ),
            self.timeseries_viewer,
        )


# Run App
graph_editor = DatasetViewer()
graph_editor.servable()
