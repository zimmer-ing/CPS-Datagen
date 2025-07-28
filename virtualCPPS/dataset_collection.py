from __future__ import annotations
from typing import List
from dataclasses import dataclass
from pathlib import Path
import pprint
import pandas as pd

import numpy as np
import torch
import holoviews as hv
import sklearn.decomposition
import sklearn.manifold
import umap

from virtualCPPS.dynamical_system import DynamicalSystem
from virtualCPPS.automaton import Automaton
from virtualCPPS.config import Config
from virtualCPPS.dataset import DataSet
from virtualCPPS.mapping_params import MappingParams


@dataclass
class DataSetCollection:
    name: str
    datasets: List[DataSet]
    mapping_params: MappingParams
    config: Config

    @classmethod
    def generate(cls, config: Config, name: str) -> cls:
        mapping_params = MappingParams.generate(config)

        data_train = DataSet.generate(
            config, mapping_params, dataset_type="train"
        )

        data_val = DataSet.generate(config, mapping_params, dataset_type="val")

        data_error = DataSet.generate(
            config, mapping_params, dataset_type="error"
        )

        dataset_collection = cls(
            name=name,
            datasets=[data_train, data_val, data_error],
            mapping_params=mapping_params,
            config=config,
        )

        if config.save_data:
            print("Saving data collection.")
            dataset_collection.save()

        if config.save_datavis:
            print("Saving data collection visualizations.")
            dataset_collection.save_datavis()

        return dataset_collection

    def __iter__(self):
        return self.datasets

    def save(self, output_path: Path = None):
        self.save_data(output_path)
        self.save_metadata(output_path)

    def save_data(
        self, output_path: Path = None,
    ):
        if output_path is None:
            output_path = Path(self.config.save_path) / Path(
                f"data_{self.config.timestamp}"
            )
        inner_output_path = output_path / Path(self.name)
        inner_output_path.mkdir(exist_ok=True, parents=True)

        for dataset in self.datasets:
            df_obs = pd.DataFrame(
                data=dataset.obs.detach().cpu().numpy(),
                columns=[
                    f"obs_{dim}" for dim in range(1, self.config.dims_obs + 1)
                ],
            )

            df_extra = pd.DataFrame(
                data=np.column_stack(
                    (
                    dataset.raw_data.cycle,
                    dataset.latent,
                    dataset.raw_data.x,
                    dataset.raw_data.y,
                    dataset.raw_data.statename
                    )
                ),
                columns=[
                    "cycle nr",
                    *[
                        f"latent_{dim}"
                        for dim in range(1, self.config.dim_latent + 1)
                    ],
                    *[
                        f"statespace_dim{dim}"
                        for dim in range(1, self.config.dim_latent + 1)
                    ],
                    "y_value",
                    'Statename',

                ],
            )
            #save extra file containing the metric in case of dynamic error
            if 'error' == dataset.dataset_type and 'dynamic'==self.config.error_type:
                df_metric=dataset.raw_data.applied_errors.cumsum()
                df_metric.to_csv(
                inner_output_path / f"dyn_error_truth.csv.gz",
                index=False,
            )

            df_obs.to_csv(
                inner_output_path / f"data_{dataset.dataset_type}.csv.gz",
                index=False,
            )

            df_extra.to_csv(
                inner_output_path / f"extra_{dataset.dataset_type}.csv.gz",
                index=False,
            )

    def save_metadata(self, output_path: Path = None):
        if output_path is None:
            output_path = Path(self.config.save_path) / Path(
                f"data_{self.config.timestamp}"
            )

        inner_output_path = output_path / Path(self.name)
        inner_output_path.mkdir(exist_ok=True, parents=True)

        filename = f"metadata.txt"
        metadata_path = inner_output_path / Path(filename)



        metadata_str = ("Config:\n"
            f"{pprint.pformat(vars(self.config))}\n"
            "\nLinear Observation Mapping Parameters:\n"
            f"{self.mapping_params.obs_lin}\n"
            "\nSinusidal Observation Mapping Parameters:\n"
            f"{self.mapping_params.obs_sin}\n"
            "\nExponential Observation Mapping Parameters:\n"
            f"{self.mapping_params.obs_exp}\n"
        )

        with metadata_path.open("w") as file:
            file.write(metadata_str)


    def save_datavis(
        self,
        output_path: Path = None,
        part: str = "latent",
        dims: int = 3,
        reducer_name: str = "pca",
    ):
        if output_path is None:
            output_path = Path(self.config.save_path) / Path(
                f"data_{self.config.timestamp}"
            )

        inner_output_path = output_path / Path(self.name)
        inner_output_path.mkdir(exist_ok=True, parents=True)

        hv_obj_latent = self.get_hv_obj("latent", dims, reducer_name)
        filename = f"vis_latent.html"
        hv.save(hv_obj_latent, inner_output_path / Path(filename))

        hv_obj_obs = self.get_hv_obj("obs", dims, reducer_name)
        filename = f"vis_obs.html"
        hv.save(hv_obj_obs, inner_output_path / Path(filename))

    def get_hv_obj(
        self, part: str = "latent", dims: int = 3, reducer_name: str = "pca",
    ):
        data_slices = {}
        slice_start = 0
        temp = []
        for dataset in self.datasets:
            if part == "latent":
                data = dataset.get_latent().cpu().numpy()
            elif part == "obs":
                data = dataset.obs.cpu().numpy()
            else:
                raise NameError("Unknown value for parameter part:", part)

            temp.append(data)

            data_slices[dataset.dataset_type] = slice(
                slice_start, slice_start + len(data)
            )
            slice_start += len(data)

        data = np.vstack(temp)

        if dims == 2:
            hv.output(backend="bokeh")
            if data.shape[1] > 2:
                reducer = self.get_reducer(reducer_name, n_components=dims)
                data = reducer.fit_transform(data)

                points_dict = {}
                for name, data_slice in data_slices.items():
                    points_dict[name] = hv.Points(
                        data[data_slice],
                        kdims=[
                            hv.Dimension(
                                f"{part}_reduced_dim_1",
                                soft_range=(
                                    -self.config.spread_cc_input / 2,
                                    self.config.spread_cc_input / 2,
                                ),
                            ),
                            hv.Dimension(
                                f"{part}_reduced_dim_2",
                                soft_range=(
                                    -self.config.spread_cc_input / 2,
                                    self.config.spread_cc_input / 2,
                                ),
                            ),
                        ],
                    )

            else:
                points_dict = {}
                for name, data_slice in data_slices.items():
                    points_dict[name] = hv.Points(
                        data[data_slice],
                        kdims=[
                            hv.Dimension(
                                f"{part}_1",
                                soft_range=(
                                    -self.config.spread_cc_input / 2,
                                    self.config.spread_cc_input / 2,
                                ),
                            ),
                            hv.Dimension(
                                f"{part}_2",
                                soft_range=(
                                    -self.config.spread_cc_input / 2,
                                    self.config.spread_cc_input / 2,
                                ),
                            ),
                        ],
                    )

            points_overlay = hv.NdOverlay(points_dict)

        elif dims == 3:
            hv.output(backend="plotly")
            if data.shape[1] > 3:
                reducer = self.get_reducer(reducer_name, n_components=dims)
                data = reducer.fit_transform(data)

                points_dict = {}
                for name, data_slice in data_slices.items():
                    points_dict[name] = hv.Scatter3D(
                        data[data_slice],
                        kdims=[
                            hv.Dimension(
                                f"{part}_reduced_dim_1",
                                soft_range=(
                                    -self.config.spread_cc_input / 2,
                                    self.config.spread_cc_input / 2,
                                ),
                            ),
                            hv.Dimension(
                                f"{part}_reduced_dim_2",
                                soft_range=(
                                    -self.config.spread_cc_input / 2,
                                    self.config.spread_cc_input / 2,
                                ),
                            ),
                        ],
                        vdims=[
                            hv.Dimension(
                                f"{part}_reduced_dim_3",
                                soft_range=(
                                    -self.config.spread_cc_output / 2,
                                    self.config.spread_cc_output / 2,
                                ),
                            ),
                        ],
                    )
            else:
                points_dict = {}
                for name, data_slice in data_slices.items():
                    points_dict[name] = hv.Scatter3D(
                        data[data_slice],
                        kdims=[
                            hv.Dimension(
                                f"{part}_1",
                                soft_range=(
                                    -self.config.spread_cc_input / 2,
                                    self.config.spread_cc_input / 2,
                                ),
                            ),
                            hv.Dimension(
                                f"{part}_2",
                                soft_range=(
                                    -self.config.spread_cc_input / 2,
                                    self.config.spread_cc_input / 2,
                                ),
                            ),
                        ],
                        vdims=[
                            hv.Dimension(
                                f"{part}_3",
                                soft_range=(
                                    -self.config.spread_cc_output / 2,
                                    self.config.spread_cc_output / 2,
                                ),
                            ),
                        ],
                    )

            points_overlay = hv.NdOverlay(points_dict)

        else:
            raise ValueError(
                "Parameter dims can ony be 2 or 3, given were:", dims
            )

        return points_overlay

    def get_reducer(self, reducer_name, n_components):
        if reducer_name == "pca":
            reducer = sklearn.decomposition.PCA(n_components=n_components)

        elif reducer_name == "tsne":
            reducer = sklearn.manifold.TSNE(n_components=n_components)

        elif reducer_name == "umap":
            reducer = umap.UMAP(n_components=n_components)

        else:
            raise NameError("Unknown reducer_name:", reducer_name)

        return reducer
