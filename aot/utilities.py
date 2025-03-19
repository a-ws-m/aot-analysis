from enum import Enum
from pathlib import Path
from typing import NamedTuple, Union

import MDAnalysis as mda
import numpy as np
import yaml

try:
    from scipy.sparse import coo_array
except ImportError:
    from scipy.sparse import coo_matrix as coo_array


def save_sparse(sparse_arrs: dict[int, coo_array], file, compressed=True):
    """Save several sparse arrays to a file."""
    arrays_dict = {}
    for frame, mat in sparse_arrs.items():
        arr_dict = {
            f"row{frame}": mat.row,
            f"col{frame}": mat.col,
            f"shape{frame}": mat.shape,
            f"data{frame}": mat.data,
        }
        arrays_dict.update(arr_dict)

    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)


def load_sparse(file) -> dict[int, coo_array]:
    """Load a sparse array from disk."""
    sparse_arrs = dict()

    with np.load(file) as loaded:
        keys = loaded.keys()
        frames = [int(key[3:]) for key in keys if key.startswith("row")]
        if not frames:
            raise ValueError("No sparse arrays found in file.")

        for frame in frames:
            row = loaded[f"row{frame}"]
            col = loaded[f"col{frame}"]
            data = loaded[f"data{frame}"]
            shape = loaded[f"shape{frame}"]

            sparse_arrs[frame] = coo_array((data, (row, col)), shape=shape)

    return sparse_arrs


class AggregateProperties(Enum):
    AGGREGATION_NUMBERS = "Aggregation numbers"
    EAB = r"$e_{ab}$"
    EAC = r"$e_{ac}$"
    RADIUS_OF_GYRATION = r"Radius of gyration ($\mathrm{\AA}$)"
    VOLUME = r"Volume ($\mathrm{\AA}^3$)"
    SURFACE_AREA = r"Surface area ($\mathrm{\AA}^2$)"
    SURFACE_AREA_PER_SURFACTANT = r"Surfactant surface area ($\mathrm{\AA}^2$)"
    SURFACE_AREA_TO_VOLUME = r"Surface area / Volume ($\mathrm{\AA}^{-1}$)"
    NORMALISED_AGGREGATION_NUMBERS = "Normalised aggregation numbers"
    TOTAL_VOLUME = r"Total excluded volume estimate ($\mathrm{\AA}^3$)"
    SOAP_SIM_1 = "Dimension 1 of KPCA of SOAP kernel"
    SOAP_SIM_2 = "Dimension 2 of KPCA of SOAP kernel"

    @classmethod
    def all(cls) -> 'set["AggregateProperties"]':
        return set(cls)

    @classmethod
    def fast(cls):
        return cls.all().difference(
            {
                cls.VOLUME,
                cls.SURFACE_AREA,
                cls.SURFACE_AREA_PER_SURFACTANT,
                cls.SURFACE_AREA_TO_VOLUME,
                cls.TOTAL_VOLUME,
            }
        )

    def __or__(self, other):
        if isinstance(other, set):
            return {self} | other
        elif isinstance(other, AggregateProperties):
            return {self, other}
        else:
            raise TypeError

    def __ror__(self, other):
        if isinstance(other, set):
            return other | {self}
        elif isinstance(other, AggregateProperties):
            return {other, self}
        else:
            raise TypeError


class Counterion(NamedTuple):
    shortname: str
    longname: str


class AtomisticResults(NamedTuple):
    """Information about some simulation results."""

    percent_aot: Union[int, float]
    counterion: Counterion
    tpr_file: Path
    traj_file: Path

    @property
    def percent_str(self) -> str:
        if isinstance(self.percent_aot, int):
            return str(self.percent_aot)
        return f"{self.percent_aot:.1f}".replace(".", "_")

    @property
    def name(self):
        return f"{self.percent_str}-{self.counterion.shortname}"

    @property
    def plot_name(self) -> str:
        return f"{self.percent_aot:.1f} wt.% AOT with {self.counterion.longname}"

    @property
    def adj_file(self) -> str:
        return f"{self.name}-adj.npz"

    @property
    def df_file(self) -> str:
        return f"{self.name}-df.csv"

    @property
    def agg_adj_file(self) -> str:
        return f"{self.name}-agg-adj.gml"

    def universe(self) -> mda.Universe:
        """Get an MDAnalysis Universe for the simulation."""
        return mda.Universe(self.tpr_file, self.traj_file)


class Coarseness(NamedTuple):
    dirname: str
    friendly_name: str
    tail_match: str
    cutoff: float


class CoarseResults(NamedTuple):
    """Information about some coarse-grained simulation results."""

    percent_aot: Union[int, float]
    coarseness: Coarseness
    tpr_file: Path
    traj_file: Path

    @property
    def tail_match(self) -> str:
        return self.coarseness.tail_match

    @property
    def cutoff(self) -> float:
        return self.coarseness.cutoff

    @property
    def percent_str(self) -> str:
        if isinstance(self.percent_aot, int):
            return str(self.percent_aot)
        return f"{self.percent_aot:.2f}".replace(".", "_")

    @property
    def name(self):
        return f"{self.coarseness.friendly_name}-{self.percent_str}"

    @property
    def plot_name(self) -> str:
        return f"{self.coarseness.friendly_name}: {self.percent_aot:.1f} wt.% AOT"

    @property
    def adj_file(self) -> str:
        return f"{self.name}-adj.npz"

    @property
    def df_file(self) -> str:
        return f"{self.name}-df.csv"

    @property
    def agg_adj_file(self) -> str:
        return f"{self.name}-agg-adj.gml"

    def universe(self) -> mda.Universe:
        """Get an MDAnalysis Universe for the simulation."""
        return mda.Universe(self.tpr_file, self.traj_file)


class ResultsYAML:

    def __init__(self, root: Path, file: str) -> None:
        self.root = root
        self.file = file
        self.path = root / file
        self.data = yaml.load(self.path.read_text(), Loader=yaml.SafeLoader)
        self._parse()

    def _parse(self):
        """Parse the incoming YAML file."""
        self.counterions = {
            key: Counterion(shortname=key, longname=val)
            for key, val in self.data["Counterions"].items()
        }

        self.atomistic_results = []
        if "AtomisticResults" in self.data:
            for res in self.data["AtomisticResults"]["results"]:
                res["percent_aot"] = float(res.pop("percent"))
                res["counterion"] = self.counterions[res["counterion"]]
                res["tpr_file"] = self.root / res["tpr_file"]
                res["traj_file"] = self.root / res["traj_file"]
                self.atomistic_results.append(AtomisticResults(**res))

        self.coarse_results = []
        if "CoarseResults" in self.data:
            for data in self.data["CoarseResults"]["results"]:
                coarseness = Coarseness(
                    data["dirname"],
                    data["friendly_name"],
                    data["tail_match"],
                    data["cutoff"],
                )

                rel_path = self.root / coarseness.dirname
                for res in data["results"]:
                    res["percent_aot"] = float(res.pop("percent"))
                    res["coarseness"] = coarseness
                    res["tpr_file"] = rel_path / res["tpr_file"]
                    res["traj_file"] = rel_path / res["traj_file"]
                    self.coarse_results.append(CoarseResults(**res))

    def get_results(self) -> "list[AtomisticResults | CoarseResults]":
        return self.atomistic_results + self.coarse_results
