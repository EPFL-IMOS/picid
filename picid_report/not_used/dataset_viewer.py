import logging
import os
from io import StringIO
from pathlib import Path

import awkward as ak
import holoviews as hv
import hydra
import numpy as np
import omegaconf
import pandas as pd
import panel as pn
import param
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.console import Console
import torch

from picid.data.preprocessing.preprocessor import PreProcessor
from picid.transforms.base.transform_manager import ConfigTransformManager
from picid.utils.omegaconf_utils import find_config_file
from picid.utils.rich_output import print_data_dict_structure
from picid.utils.torch_utils import correct_state_dict_keys

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
    dataset_source = instantiate(cfg.datasource)

    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    preprocessor = PreProcessor(datasource=dataset_source, transforms=transforms_manager)

    # Get cache dirs and flags
    data_cache_path = (
        cfg.paths.cache_path
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    data_library_part_path = (
        Path(cfg.paths.root_dir) / "picid/data/datasources"
        if cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
        else None
    )
    transform_library_part_path = (
        Path(cfg.paths.root_dir) / "picid/transforms"
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
# Time Series Viewer Component


class TimeSeriesViewer(pn.viewable.Viewer):

    train_val_test_selector = param.Selector(
        objects=["train", "val", "test"], default="train"
    )

    data = param.Dict(default={})
    model = param.Parameter()
    split_data: ak.Array = param.Parameter()
    unit = param.Integer(default=0, bounds=(0, None))
    segment = param.Integer(default=0, bounds=(0, None))
    zoom = param.Range(default=(0, 0))
    target = param.Parameter(default="")

    def __init__(self, **params):
        super().__init__(**params)

        # Persistent plot pane
        self.plot_pane = pn.pane.HoloViews(hv.Curve([]), sizing_mode="stretch_both")

        self.unit_slider = pn.widgets.IntSlider.from_param(self.param.unit)
        self.segment_slider = pn.widgets.IntSlider.from_param(self.param.segment)
        self.zoom_slider = pn.widgets.RangeSlider.from_param(self.param.zoom)

        # watch model parameters, not widgets
        self.param.watch(self._data_updated, ["data", "train_val_test_selector"])
        self.param.watch(self._update_unit, ["unit", "split_data"])
        self.param.watch(self._update_segment, ["unit", "segment", "split_data"])
        self.param.watch(self._update_plot, ["unit", "segment", "split_data"])
        self.param.watch(self._update_plot_zoom, ["zoom"])

    def _data_updated(self, event):
        print("data updated event triggered")
        data = event.obj.data
        if not data:
            return
        print("Data updated, updating widgets...")
        split_data = data.get(self.train_val_test_selector, {})
        n_units = len(split_data.get("features", []))

        self.unit_slider.end = max(n_units - 1, 0)
        self.split_data = split_data

    def _update_unit(self, event):
        if not self.split_data:
            return

        unit = self.unit
        features = self.split_data["features"][unit]

        if isinstance(features, np.ndarray):
            assert features.ndim == 2, "Expected 2D array (time x features)"
            n_segments = 1
        elif isinstance(features, ak.Array):
            assert features.ndim == 3, "Expected 3D array (segments x time x features)"
            n_segments = len(features)
        else:
            raise ValueError(f"Unsupported data type: {type(features)}")

        self.segment_slider.name = f"Segment (total: {n_segments})"
        self.segment = 0
        self.segment_slider.start = 0
        self.segment_slider.end = max(n_segments - 1, 0)

    def _update_segment(self, event):
        print("_update_segment triggered")
        segment = self.segment
        unit = self.unit
        features = self.split_data["features"][unit]

        if isinstance(features, np.ndarray):
            assert features.ndim == 2, "Expected 2D array (time x features)"
            features = features[np.newaxis, ...]  # add segment axis
            # convert pandas to numpy if needed
        elif isinstance(features, ak.Array):
            assert features.ndim == 3, "Expected 3D array (segments x time x features)"
        else:
            raise ValueError(f"Unsupported data type: {type(features)}")
        features = features[segment]

        self.zoom = (0, len(features) - 1)
        self.zoom_slider.start = 0
        self.zoom_slider.end = len(features) - 1

    def _get_plot(self, unit=None, segment=None, xlim=None):
        if self.data is None or not self.split_data:
            return hv.Curve([])

        if unit is None:
            unit = self.unit_slider.value
        if segment is None:
            segment = self.segment_slider.value

        plot_data = self.split_data["features"][unit]
        target = self.split_data["target"][unit]

        if isinstance(plot_data, np.ndarray):
            assert plot_data.ndim == 2, "Expected 2D array (time x features)"
            plot_data = plot_data[np.newaxis, ...]  # add segment axis
            if isinstance(target, pd.Series):
                target = target.to_numpy()
            target = target[np.newaxis, ...]
        elif isinstance(plot_data, ak.Array):
            assert plot_data.ndim == 3, "Expected 3D array (segments x time x features)"
        else:
            raise ValueError(f"Unsupported data type: {type(plot_data)}")

        plot_data = plot_data[segment]
        plot_data = ak.to_numpy(plot_data)
        target = target[segment]

        df = pd.DataFrame(
            plot_data[:128, :],
            columns=[f"feat_{i}" for i in range(plot_data.shape[1])],
        )

        # torch.Size([1, 212, 3])
        if self.model is not None:
            plot_data_torch = torch.from_numpy(plot_data).unsqueeze(0).float()
            with torch.no_grad():
                recon_torch = self.model.backbone.forward(plot_data_torch[:, :128, :])
                recon = recon_torch.squeeze(0).numpy()  # shape (128, features)
                recon_df = pd.DataFrame(
                    recon, columns=[f"recon_{i}" for i in range(recon.shape[1])]
                )

            df = pd.concat([df, recon_df], axis=1)

        if len(df) > 100000:
            print(f"Downsampling data for performance... (original length: {len(df)})")
            df = df.iloc[::100, :]  # downsample for performance

        # matplotlib default 10-color cycle
        mpl_cycle = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        plot = hv.Overlay(
            [
                df.hvplot.line(
                    y=f"feat_{i}",
                    line_dash="dashed",
                    color=mpl_cycle[i % len(mpl_cycle)],
                    width=600,
                    height=300,
                    label=f"feat_{i}",
                )
                * df.hvplot.line(
                    y=f"recon_{i}",
                    line_dash="solid",
                    color=mpl_cycle[i % len(mpl_cycle)],
                    width=600,
                    height=300,
                    label=f"recon_{i}",
                )
                for i in range(plot_data.shape[1])
            ]
        ).opts(legend_position="right", show_legend=True)

        self.target = target

        if xlim is not None:
            plot = plot.opts(xlim=xlim)
        return plot

    def _update_plot_zoom(self, event):
        print("_update_plot_zoom triggered")
        self.plot_pane.object = self._get_plot(xlim=event.new)

    def _update_plot(self, event):
        print("_update_plot triggered")
        self.plot_pane.object = self._get_plot()

    def get_target(self):
        target = self.target
        if len(target) == 1:
            return str(target[0])
        if isinstance(target, np.ndarray):
            df = pd.DataFrame(target)
            if len(df) > 100000:
                print(
                    f"Downsampling target for performance... (original length: {len(df)})"
                )
                df = df.iloc[::100, :]  # downsample for performance
            plot = df.hvplot()
            return pn.pane.HoloViews(plot, sizing_mode="stretch_both")
        return str(target)

    @param.depends("split_data", watch=True)
    def controls_panel(self):
        return pn.Row(
            pn.Param(self, parameters=["train_val_test_selector", "use_model"]),
            self.unit_slider if self.unit_slider else pn.pane.Markdown("_No data yet_"),
            (
                self.segment_slider
                if self.segment_slider
                else pn.pane.Markdown("_No data yet_")
            ),
            self.zoom_slider if self.zoom else pn.pane.Markdown("_No data yet_"),
        )

    def __panel__(self):
        return pn.Column(
            self.controls_panel,
            pn.Row(
                pn.pane.Markdown("**Target**"),
                self.get_target,
                height=200,
                sizing_mode="stretch_width",
            ),
            pn.Row(self.plot_pane),
        )


def parse_directory_for_model_checkpoints(directory: Path):
    """Parse a directory and its subdirectories for *.pt model checkpoints and return a list of their full paths."""
    model_checkpoints = []
    if not directory.exists() or not directory.is_dir():
        logger.warning(f"Directory {directory} does not exist or is not a directory.")
        return model_checkpoints

    for item in directory.rglob("*.ckpt"):  # Recursively search for .ckpt files
        if item.is_file():
            model_checkpoints.append(item.as_posix())  # Add full path as string
        else:
            logger.warning(f"Found {item} but it is not a file.")
    return model_checkpoints


class DatasetViewer(pn.viewable.Viewer):

    # Model Params
    model = param.Parameter()
    model_selector = param.Selector(objects=[], default=None)
    use_model = param.Boolean(default=True)

    # Dataset Params
    dataset_selector = param.Selector(objects=[], default=None)
    load_dataset_button = param.Action(
        lambda x: x.param.trigger("load_dataset_button"), label="Load Dataset"
    )

    debug = param.Boolean(default=False, label="Debug Mode")
    cfg = param.Parameter()

    def __init__(self, **params):
        super().__init__(**params)

        experiment_aliases = {
            "nb14/reconstruction/residual_autoencoder": "reconstruction_nb14",
            "pronostia/reconstruction/residual_autoencoder": "reconstruction_nb14",
            "xjtu_sy/reconstruction/residual_autoencoder": "reconstruction_nb14",
            "unibo/reconstruction/residual_autoencoder": "reconstruction_nb14",
        }

        artifacts_path = Path("artifacts").absolute()
        logger.info(f"Parsing model checkpoints in {artifacts_path}")

        ckpts = parse_directory_for_model_checkpoints(Path("artifacts"))
        self.param.model_selector.objects = ckpts
        self.model_selector = ckpts[0]

        # update the parameter objects (not overwrite)
        self.param.dataset_selector.objects = list(experiment_aliases.keys())
        self.dataset_selector = list(experiment_aliases.keys())[0]  # set default value

        self.data_structure_description = pn.Column("No data loaded yet")
        self.data_dict = {}
        self.timeseries_viewer = TimeSeriesViewer()

    def load_config(self, exp_alias: str) -> OmegaConf:

        def load_cfg(overrides):
            with initialize(version_base="1.2", config_path="../configs"):
                cfg = compose(config_name="run.yaml", overrides=overrides)
                return cfg

        debug_cfg_dict = {
            "nb14/reconstruction/residual_autoencoder": "debug=nb14",
            "pronostia/reconstruction/residual_autoencoder": "debug=default",
            "xjtu_sy/reconstruction/residual_autoencoder": "debug=xjtu",
            "unibo/reconstruction/residual_autoencoder": "debug=unibo",
        }

        debug_cfg = [debug_cfg_dict.get(exp_alias)] if self.debug else []
        overrides = [f"experiment={exp_alias}", "paths=rt_local", *debug_cfg]
        logger.info(f"Loading config with args: {overrides}")
        cfg = load_cfg(overrides)

        logger.info(f"Loading cfg with alias: {exp_alias}")
        return cfg

    @param.depends("load_dataset_button", watch=True)
    def _on_load_dataset_button_pressed(self):
        if self.cfg is None:
            logger.warning("No cfg available, skipping dataset load")
            return

        self.data_dict = load_dataset_from_cfg(self.cfg)
        # print("Loading dataset...")
        # with open("testing_data.pkl", "rb") as f:
        #     self.data_dict = pickle.load(f)

        if self.use_model:
            self.timeseries_viewer.model = self.model

        self.timeseries_viewer.data = self.data_dict
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, color_system=None)
        tree = print_data_dict_structure(self.data_dict)
        console.print(tree)
        self.data_structure_description.clear()
        self.data_structure_description.append(pn.pane.Str(buf.getvalue(), width=600))

    @param.depends("dataset_selector", "debug", watch=True)
    def _update_cfg(self):
        """Display the configuration of the selected dataset."""
        if self.cfg is not None:
            self.data_structure_description.clear()
        cfg = self.load_config(self.dataset_selector)
        self.cfg = cfg  # Store cfg for later use

    @param.depends("model_selector", "use_model", watch=True)
    def _load_model(self):
        """Loads the model from the specified experiment directory."""
        if self.model_selector is None or not self.use_model:
            logger.warning("No model selected, skipping model load")
            self.model = None
            return

        logger.info(
            f"Loading model from: {self.model_selector}, path exists: {Path(self.model_selector).exists()}"
        )

        config_path = find_config_file(
            Path(self.model_selector).parent.parent.parent,
            config_name="config_resolved.yaml",
            select_from_hydra=False,
        )
        cfg = omegaconf.OmegaConf.load(config_path)
        model = hydra.utils.instantiate(cfg.model)

        state_dict = torch.load(
            self.model_selector, map_location="cpu", weights_only=False
        )["state_dict"]

        # Correct the keys in the state_dict if necessary
        corrected_state_dict = correct_state_dict_keys(state_dict)
        model.load_state_dict(corrected_state_dict)
        logger.info(f"Model loaded successfully from {self.model_selector}")
        self.model = model

    @param.depends("cfg")
    def config_pane(self):
        if self.cfg is None:
            logger.warning("No cfg available, skipping dataset load")
            return

        cfg_json = OmegaConf.to_container(self.cfg.datasource, resolve=True)
        json_pane = pn.pane.JSON(cfg_json, name="Configuration", depth=5, width=600)
        logger.info(f"Loaded config for dataset: {self.dataset_selector}")
        return json_pane

    def model_info_pane(self):
        if self.model is None:
            return pn.pane.Markdown("No model loaded")

        model_str = repr(self.model)
        return pn.Column(
            pn.pane.Markdown(f"```\n{model_str}\n```", width=600),
            scroll=True,
            height=300,
        )

    def __panel__(self):
        """Panel layout structure."""
        return pn.Row(
            pn.Column(
                pn.Param(self.param),
                self.config_pane,
                self.data_structure_description,
                self.model_info_pane,
            ),
            self.timeseries_viewer,
        )


# Run App
graph_editor = DatasetViewer()
graph_editor.servable()
