"""Utilities for finding cluster sizes across simulations.

TODO:
    * Complete configuration of :class:`AggregateProperties` so that we can
      check whether a saved `DataFrame` has all the necessary columns.
    * Implement full CLI functionality and allow `SimResults` and
      `CoarseSimResults` to be stored in a JSON file.

"""

import argparse
from enum import Enum
from functools import lru_cache

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from pathlib import Path
from typing import NamedTuple, Optional, Union

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytim
import pyvista as pv
import seaborn as sns
import yaml
from MDAnalysis.analysis.base import AnalysisBase, AtomGroup
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.core.groups import ResidueGroup
from pytim.datafiles import CHARMM27_TOP, pytim_data
from pytim.interface import Interface
from scipy.spatial.distance import pdist

try:
    from scipy.sparse import coo_array
except ImportError:
    from scipy.sparse import coo_matrix as coo_array

from scipy.sparse.csgraph import connected_components

TIME_COL = "Time (ns)"

EAB = "$e_{ab}$"
EAC = "$e_{ac}$"


def center_on_cluster(cluster: AtomGroup):
    """Center the universe on a specific cluster to avoid PBC issues."""
    Interface.center_system("spherical", cluster, None)


def atom_to_mol_pairs(atom_pairs: np.ndarray, atom_per_mol: int) -> np.ndarray:
    """Convert an array of atom pairs to an array of molecule pairs."""
    # These are already sorted, so a floor division will give us the molecule idxs
    mol_pairs = (atom_pairs // atom_per_mol).astype(np.int32)
    mol_pairs = np.unique(mol_pairs, axis=0)

    # Get rid of self-connections
    mask = np.where(mol_pairs[:, 0] != mol_pairs[:, 1])

    return mol_pairs[mask]


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


def hydrodynamic_radius(group: AtomGroup) -> float:
    """Calculate the hydrodynamic radius for a given group of atoms."""
    center_on_cluster(group)
    return 1 / ((1 / len(group) ** 2) * np.sum(1 / pdist(group.positions)))


def radius_of_gyration(group: AtomGroup) -> float:
    """Calculate the radius of gyration for a given group of atoms."""
    center_on_cluster(group)
    return group.radius_of_gyration()


def willard_chandler(
    group: AtomGroup, radii_dict: dict = pytim_data.vdwradii(CHARMM27_TOP)
) -> "tuple[float, float]":
    """Calculate the Willard-Chandler surface for a given group of atoms and associated properties."""
    u = group.universe

    # Get the surface
    wc = pytim.WillardChandler(
        u,
        group=group,
        alpha=4.0,
        mesh=2.0,
        radii_dict=radii_dict,
        autoassign=False,
        density_cutoff_ratio=0.33,
        centered=True,
    )

    # radius, _, _, _ = pytim.utilities.fit_sphere(wc.triangulated_surface[0])

    # converting PyTim to PyVista surface
    verts = wc.triangulated_surface[0]
    faces = wc.triangulated_surface[1]
    threes = 3 * np.ones((faces.shape[0], 1), dtype=int)
    faces = np.concatenate((threes, faces), axis=1)
    poly = pv.PolyData(verts, faces)

    # Get actual surface volume
    volume = poly.volume
    surf = poly.area

    return volume, surf


def get_cpe(atoms: AtomGroup):
    """
    Compute coordinate pair eccentricities for a given set of semi-axes.
    Returns 2 values, eab and eac, both on [0,1]
    """
    center_on_cluster(atoms)

    # moments_val for the MoI themselves, princ_vec for the vector directions
    moments_val, princ_vec = np.linalg.eig(atoms.moment_of_inertia())

    # getting mass of the selected group of atoms
    mass = atoms.total_mass()

    # sortinng MoI and vectors by size of MoI
    idx = moments_val.argsort()[::-1]

    moments_val = moments_val[idx]
    princ_vec = princ_vec[:, idx]

    # Array to solve for axis lengths
    inverter = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * 0.5

    # converting MoI to axis length eigenvalues
    semiaxes = np.sqrt((5 / mass) * (np.matmul(inverter, np.transpose(moments_val))))
    c, b, a = np.sort(semiaxes)

    eab = np.sqrt(1 - (b**2 / a**2))
    eac = np.sqrt(1 - (c**2 / a**2))

    return eab, eac


def get_adj_array(
    tailgroups: AtomGroup, cutoff: float, box_dim: np.ndarray
) -> coo_array:
    """Calculate the adjacency matrix for the current frame."""
    atom_pairs = capped_distance(
        tailgroups,
        tailgroups,
        cutoff,
        box=box_dim,
        return_distances=False,
    )

    num_surf = tailgroups.n_residues
    atom_per_mol = int(len(tailgroups) / num_surf)

    mol_pairs = atom_to_mol_pairs(atom_pairs, atom_per_mol)
    num_pairs = mol_pairs.shape[0]
    ones = np.ones((num_pairs,), dtype=np.bool_)
    return coo_array(
        (ones, (mol_pairs[:, 0], mol_pairs[:, 1])),
        shape=(num_surf, num_surf),
        dtype=np.bool_,
    )


class AggregateProperties(Enum):
    AGGREGATION_NUMBERS = "Aggregation numbers"
    EAB = "Eab"
    EAC = "Eac"
    RADIUS_OF_GYRATION = "Radius of gyration"
    VOLUME = "Volume"
    SURFACE_AREA = "Surface area"
    HYDRODYNAMIC_RADIUS = "Hydrodynamic radius"

    @classmethod
    def all(cls) -> 'set["AggregateProperties"]':
        return set(cls)

    @classmethod
    def fast(cls):
        return cls.all().difference(
            {
                cls.VOLUME,
                cls.SURFACE_AREA,
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


class MicelleAdjacency(AnalysisBase):
    """Class for computing the adjacency matrix of surfactants in micelles.

    Notes:

        * If you are updating a `current_df` you must use the same `min_cluster_size` as before!

    """

    def __init__(
        self,
        tailgroups: AtomGroup,
        cutoff: float = 4.25,
        min_cluster_size: int = 5,
        properties: "set[AggregateProperties]" = AggregateProperties.all(),
        coarse: bool = False,
        verbose=True,
        current_df: Optional[pd.DataFrame] = None,
        current_adj_mats: Optional[dict[int, coo_array]] = None,
        **kwargs,
    ):
        """Split tails into different molecules."""
        trajectory = tailgroups.universe.trajectory
        super().__init__(trajectory, verbose, **kwargs)

        self.cutoff = cutoff
        self.min_cluster_size = min_cluster_size
        self.properties = properties
        self.coarse = coarse

        self.num_surf: int = tailgroups.n_residues
        self.whole_molecules: ResidueGroup = tailgroups.residues.unique

        # Sort the tailgroups by residue number
        tailgroups_ = tailgroups.universe.atoms[[]]
        for residue in self.whole_molecules:
            tailgroups_ += residue.atoms & tailgroups
        self.tailgroups = tailgroups_
        assert len(self.tailgroups) == len(tailgroups)

        self.atom_per_mol = int(len(self.tailgroups) / self.num_surf)

        # Load any data we already have
        self.df = current_df if current_df is not None else pd.DataFrame({"Frame": []})
        self.adj_mats: dict[int, coo_array] = (
            current_adj_mats if current_adj_mats is not None else dict()
        )

    @cached_property
    def vdwradii(self) -> "dict[str, float]":
        """Determine the van der Waals radii for the atoms in the system."""
        if self.coarse:
            radii = dict()
            for atom in self.tailgroups.universe.atoms:
                # See Section 8.2 of
                # https://cgmartini.nl/docs/tutorials/Martini3/Small_Molecule_Parametrization/#molecular-volume-and-shape
                atom_type = atom.type[0]
                if atom_type == "S":
                    radius = 0.230
                elif atom_type == "T":
                    radius = 0.191
                else:
                    radius = 0.264

                radii[atom.name] = radius

            return radii

        else:
            return pytim_data.vdwradii(CHARMM27_TOP)

    def do_calculate(
        self,
        properties: AggregateProperties | set[AggregateProperties],
        current_entry: Optional[pd.Series],
    ) -> bool:
        """Check whether we need to calculate the given properties, or whether they're in the DataFrame."""
        prop_set = (
            {properties} if isinstance(properties, AggregateProperties) else properties
        )
        in_properties_to_calc = bool(self.properties.intersection(prop_set))

        not_in_entry = current_entry is None or any(
            pd.isna(current_entry.get(prop.value)) for prop in prop_set
        )

        return in_properties_to_calc and not_in_entry

    def _prepare(self):
        """Initialise the results."""
        self.frame_counter: list[int] = []
        self.time_counter: list[int] = []
        self.agg_nums: list[int] = []

        self.volume: list[float] = []
        self.surface: list[float] = []

        # Coordinate pair eccentricities
        self.eabs: list[float] = []
        self.eacs: list[float] = []

        self.radii_of_gyration: list[float] = []
        self.hydrodynamic_radius: list[float] = []

    def _single_frame(self):
        """Calculate the contact matrix for the current frame."""
        current_frame = int(self._ts.frame)

        try:
            sparse_adj_arr = self.adj_mats[current_frame]
        except KeyError:
            sparse_adj_arr = get_adj_array(
                self.tailgroups, self.cutoff, self._ts.dimensions
            )
            self.adj_mats[current_frame] = sparse_adj_arr

        n_aggregates, connected_comps = connected_components(
            sparse_adj_arr, directed=False
        )

        current_frame_entries = self.df.loc[self.df["Frame"] == current_frame]
        agg_idx = 0

        for i in range(n_aggregates):
            agg_residues: mda.ResidueGroup = self.whole_molecules[
                np.where(connected_comps == i)
            ]
            agg_num = len(agg_residues)

            if agg_num < self.min_cluster_size:
                continue

            try:
                current_agg_entry = current_frame_entries.iloc[agg_idx]
                current_idx = int(current_frame_entries.index.values[0]) + agg_idx
            except IndexError:
                current_agg_entry = None
                current_idx = None
                # We'll need to add new entries to the DataFrame

            if self.do_calculate(
                AggregateProperties.VOLUME | AggregateProperties.SURFACE_AREA,
                current_agg_entry,
            ):
                vol, surf = willard_chandler(agg_residues.atoms, self.vdwradii)
                if current_idx is None:
                    self.volume.append(vol)
                    self.surface.append(surf)
                else:
                    self.df.loc[current_idx, AggregateProperties.VOLUME.value] = vol
                    self.df.loc[current_idx, AggregateProperties.SURFACE_AREA.value] = (
                        surf
                    )

            if self.do_calculate(
                AggregateProperties.EAB | AggregateProperties.EAC, current_agg_entry
            ):
                eab, eac = get_cpe(agg_residues.atoms)
                if current_idx is None:
                    self.eabs.append(eab)
                    self.eacs.append(eac)
                else:
                    self.df.loc[current_idx, AggregateProperties.EAB.value] = eab
                    self.df.loc[current_idx, AggregateProperties.EAC.value] = eac

            if self.do_calculate(
                AggregateProperties.RADIUS_OF_GYRATION, current_agg_entry
            ):
                if current_idx is None:
                    self.radii_of_gyration.append(
                        radius_of_gyration(agg_residues.atoms)
                    )
                else:
                    self.df.loc[
                        current_idx, AggregateProperties.RADIUS_OF_GYRATION.value
                    ] = radius_of_gyration(agg_residues.atoms)

            if self.do_calculate(
                AggregateProperties.HYDRODYNAMIC_RADIUS, current_agg_entry
            ):
                if current_idx is None:
                    self.hydrodynamic_radius.append(
                        hydrodynamic_radius(agg_residues.atoms)
                    )
                else:
                    self.df.loc[
                        current_idx, AggregateProperties.HYDRODYNAMIC_RADIUS.value
                    ] = hydrodynamic_radius(agg_residues.atoms)

            if current_idx is None:
                self.agg_nums.append(agg_num)

                self.frame_counter.append(self._ts.frame)
                self.time_counter.append(self._ts.time)

            # TODO: This is why the min_cluster_size must be the same when updating.
            # TODO: This index doesn't keep track of how many aggregates below the threshold were skipped
            agg_idx += 1

    def _conclude(self):
        """Store results in DataFrame."""
        data = {
            "Frame": self.frame_counter,
            "Time (ps)": self.time_counter,
            AggregateProperties.AGGREGATION_NUMBERS.value: self.agg_nums,
            "Normalised aggregation numbers": np.array(self.agg_nums) / self.num_surf,
            AggregateProperties.EAB.value: self.eabs,
            AggregateProperties.EAC.value: self.eacs,
            AggregateProperties.RADIUS_OF_GYRATION.value: self.radii_of_gyration,
            AggregateProperties.VOLUME.value: self.volume,
            AggregateProperties.SURFACE_AREA.value: self.surface,
            AggregateProperties.HYDRODYNAMIC_RADIUS.value: self.hydrodynamic_radius,
        }
        data = {key: val for key, val in data.items() if len(val)}

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)
        self.df.sort_values("Frame", inplace=True, ignore_index=True)

    def save(self, adj_path: Path, df_path: Path):
        save_sparse(self.adj_mats, adj_path)
        self.df.to_csv(df_path)


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


def all_atomistic_ma(
    result: AtomisticResults,
    min_cluster_size: int = 5,
    step: int = 1,
    properties: "set[AggregateProperties]" = AggregateProperties.fast(),
    current_df: Optional[pd.DataFrame] = None,
    current_adj_mats: Optional[dict[int, coo_array]] = None,
) -> MicelleAdjacency:
    """Run a micelle adjacency analysis for the default tail group indices."""
    u = result.universe()

    # Find the atoms in tail groups
    tail_atom_nums = [6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20]
    tail_atom_names = [f"C{idx}" for idx in tail_atom_nums]
    sel_str = " or ".join([f"name {name}" for name in tail_atom_names])
    end_atoms = u.select_atoms(sel_str)

    ma = MicelleAdjacency(
        end_atoms,
        min_cluster_size=min_cluster_size,
        properties=properties,
        current_df=current_df,
        current_adj_mats=current_adj_mats,
    )
    ma.run(step=step)

    return ma


def coarse_ma(
    result: CoarseResults,
    min_cluster_size: int = 5,
    step: int = 1,
    properties: "set[AggregateProperties]" = AggregateProperties.fast(),
    current_df: Optional[pd.DataFrame] = None,
    current_adj_mats: Optional[dict[int, coo_array]] = None,
) -> MicelleAdjacency:
    """Run a micelle adjacency analysis for the default tail group indices."""
    u = result.universe()

    tail_atoms = u.select_atoms(result.tail_match)

    ma = MicelleAdjacency(
        tail_atoms,
        cutoff=result.cutoff,
        min_cluster_size=min_cluster_size,
        coarse=True,
        properties=properties,
        current_df=current_df,
        current_adj_mats=current_adj_mats,
    )
    ma.run(step=step)

    return ma


def batch_ma_analysis(
    results: "list[AtomisticResults | CoarseResults]",
    min_cluster_size: int = 5,
    step: int = 1,
    only_last: bool = False,
    dir_: Path = Path("."),
    overwrite: bool = False,
    properties: "set[AggregateProperties]" = AggregateProperties.fast(),
) -> pd.DataFrame:
    """Load MA results from disk or run analyses anew."""
    plot_df = pd.DataFrame()
    for result in results:
        adj_path = dir_ / result.adj_file
        df_path = dir_ / result.df_file

        this_df = None
        current_adj_mats = None

        if not overwrite:
            if df_path.exists():
                print(f"Found existing DataFrame for {result.plot_name}.")
                this_df = pd.read_csv(df_path, index_col=0)
            if adj_path.exists():
                print(f"Found existing adjacency matrix for {result.plot_name}.")
                current_adj_mats = load_sparse(adj_path)

        print(f"Analysing {result.plot_name} results.")

        if isinstance(result, CoarseResults):
            ma = coarse_ma(
                result,
                min_cluster_size,
                step,
                properties=properties,
                current_df=this_df,
                current_adj_mats=current_adj_mats,
            )
        else:
            ma = all_atomistic_ma(
                result,
                min_cluster_size,
                step,
                properties=properties,
                current_df=this_df,
                current_adj_mats=current_adj_mats,
            )
        ma.save(adj_path, df_path)

        this_df = ma.df

        if only_last:
            this_df = this_df.loc[this_df["Frame"] == this_df["Frame"].max()]

        this_df = this_df.loc[this_df["Aggregation numbers"] >= min_cluster_size]

        this_df["% AOT"] = f"{result.percent_aot:.1f}"

        this_df["% AOT"] = this_df["% AOT"].astype("category")
        this_df["Simulation"] = result.plot_name
        if isinstance(result, AtomisticResults):
            this_df["Type"] = result.counterion.longname
        else:
            this_df["Type"] = result.coarseness.friendly_name

        plot_df = pd.concat([plot_df, this_df], ignore_index=True)

    return plot_df


@lru_cache
def load_results_datasets(
    results: tuple[AtomisticResults | CoarseResults, ...],
    min_cluster_size: int = 5,
    dir_: Path = Path("."),
) -> pd.DataFrame:
    """Load results datasets for plotting."""
    plot_df = pd.DataFrame()
    for result in results:
        df_path = dir_ / result.df_file

        if not df_path.exists():
            raise FileNotFoundError(f"Could not find DataFrame for {result.plot_name}.")
        else:
            print(f"Found existing files for {result.plot_name}.")
            this_df = pd.read_csv(df_path, index_col=0)

        this_df = this_df[this_df["Aggregation numbers"] >= min_cluster_size]
        this_df = this_df.groupby("Time (ps)").mean()
        this_df["Time (ps)"] = this_df.index
        this_df["% AOT"] = result.percent_aot
        this_df["Simulation"] = result.plot_name
        this_df["Type"] = (
            result.counterion.longname
            if isinstance(result, AtomisticResults)
            else result.coarseness.friendly_name
        )

        plot_df = pd.concat([plot_df, this_df], ignore_index=True)

    plot_df[TIME_COL] = plot_df["Time (ps)"] * 1e-3
    plot_df["Log agg. num"] = plot_df["Aggregation numbers"].apply(np.log10)
    plot_df["Norm. agg. number"] = plot_df["Normalised aggregation numbers"]

    try:
        plot_df[EAB] = plot_df["Eab"]
        plot_df[EAC] = plot_df["Eac"]
    except KeyError:
        pass

    try:
        plot_df["Surfactant surface area"] = (
            plot_df["Surface area"] / plot_df["Aggregation numbers"]
        )
        plot_df["Surface area / Volume"] = plot_df["Surface area"] / plot_df["Volume"]
    except KeyError:
        pass

    return plot_df


def compare_val(
    results: list[AtomisticResults | CoarseResults],
    graph_file: Path,
    y_axis: str,
    min_cluster_size: int = 5,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = load_results_datasets(tuple(results), min_cluster_size)

    print("Done analysing results!")
    print("Plotting graphs.")

    g = sns.relplot(
        data=plot_df,
        x=TIME_COL,
        y=y_axis,
        row="% AOT",
        col="Type",
        hue="Type",
        kind="line",
        errorbar="sd",
        # margin_titles=True,
        # sharey="row",
        facet_kws={"margin_titles": True, "despine": False, "sharey": "row"},
    )
    g.tight_layout()
    g.savefig(graph_file, transparent=True)


def compare_dist(
    results: list[AtomisticResults | CoarseResults],
    graph_file: Path,
    y_axis: str,
    use_interval: bool = True,
    interval: int = 50,
    min_cluster_size: int = 5,
    ylim: Optional["tuple[float, float]"] = None,
    semilog: bool = False,
    hue="Norm. agg. number",
    rename: Optional[str] = None,
    marker: str = "P",
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = load_results_datasets(tuple(results), min_cluster_size)

    if use_interval:
        plot_df = plot_df[plot_df["Frame"] % interval == 0]

    if rename is not None:
        plot_df[rename] = plot_df[y_axis]

    print("Done analysing results!")
    print("Plotting graphs.")

    g = sns.catplot(
        data=plot_df,
        x=TIME_COL,
        # order=time_labels,
        y=y_axis if not rename else rename,
        row="% AOT",
        col="Type",
        hue=hue,
        native_scale=True,
        kind="strip",
        # inner=None,
        sharey=True,
        margin_titles=True,
        # facet_kws={"margin_titles": True, "despine": False},
        palette="flare",
        marker=marker,
    )
    # g.map_dataframe(
    #     sns.swarmplot,
    #     color="k",
    #     size=3,
    #     y=y_axis,
    #     x=TIME_COL,
    # )
    if semilog:
        g.set(yscale="log")
    if ylim is not None:
        g.set(ylim=ylim)

    # g.set_xticklabels(time_labels, rotation=45)
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def compare_cpe(
    results: list[AtomisticResults | CoarseResults],
    graph_file: Path,
    use_interval: bool = False,
    interval: int = 50,
    min_cluster_size: int = 5,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = load_results_datasets(tuple(results), min_cluster_size)

    if use_interval:
        plot_df = plot_df[plot_df["Frame"] % interval == 0]

    print("Done analysing results!")
    print("Plotting graphs.")

    g = sns.relplot(
        kind="scatter",
        data=plot_df,
        x=EAB,
        y=EAC,
        row="% AOT",
        col="Type",
        hue="Norm. agg. number",
        # margin_titles=False,
        facet_kws={"margin_titles": True, "despine": False},
        palette="flare",
    )
    for ax in g.axes.flatten():
        grid_col = plt.rcParams["grid.color"]
        ax.plot([0, 1], [0, 1], c=grid_col, lw=2, linestyle="--", zorder=-0.5)
    g.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def compare_clustering(
    results: "list[AtomisticResults | CoarseResults]",
    graph_file: Path,
    min_cluster_size: int = 5,
    dir_: Path = Path("."),
):
    """Compare the clustering behaviour of several simulations."""

    plot_df = load_results_datasets(tuple(results), min_cluster_size, dir_)

    y_vars = plot_df.columns
    plot_dfm = plot_df.melt(
        id_vars=["Simulation", TIME_COL, "% AOT"],
        value_vars=list(y_vars.drop(TIME_COL)),
    )

    plot_dfm["% AOT"] = plot_dfm["% AOT"].astype("category")

    g = sns.relplot(
        data=plot_dfm,
        x=TIME_COL,
        y="value",
        col="Simulation",
        hue="% AOT",
        row="variable",
        kind="line",
        errorbar="sd",
        # margin_titles=True,
        # sharey="row",
        facet_kws={"margin_titles": True, "despine": False, "sharey": "row"},
    )

    g.figure.suptitle(
        f"Clustering comparison with min cluster size = {min_cluster_size}"
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels(TIME_COL, "Value")
    g.tight_layout()
    g.savefig(graph_file)


def tail_rdf(results: "list[CoarseResults]", graph_file: Path, step=10, start=0):
    """Plot the radial distribution between tail group beads."""
    plot_df = pd.DataFrame()

    for result in results:
        u = result.universe()
        tail_atoms = u.select_atoms(result.tail_match)
        rdf = InterRDF(
            tail_atoms,
            tail_atoms,
            range=(2.5, 7),
            nbins=100,
            exclude_same="residue",
        )
        rdf.run(verbose=True, step=step, start=start)

        df = pd.DataFrame(
            {r"Distance ($\AA$)": rdf.results.bins, r"$g(r)$": rdf.results.rdf}
        )
        df["Mapping"] = result.coarseness.friendly_name
        df["% AOT"] = str(result.percent_aot)
        plot_df = pd.concat([plot_df, df], ignore_index=True)

    g = sns.relplot(
        data=plot_df,
        x=r"Distance ($\AA$)",
        y=r"$g(r)$",
        col="Mapping",
        kind="line",
        hue="% AOT",
        # margin_titles=True,
        facet_kws={"margin_titles": True, "despine": False},
    )
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def main():
    """Commandline interface for program."""
    sns.set_theme(context="talk", palette="flare")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default=".",
        help="Directory to look in for files.",
    )
    parser.add_argument(
        "-r",
        type=str,
        default="results.yaml",
        help="YAML file containing the results to analyse. See tests/testfiles/results.yaml for an example.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=100,
        dest="step_size",
        help="Number of steps to skip in the trajectory.",
    )
    parser.add_argument(
        "-s",
        type=int,
        default=0,
        dest="start",
        help="Start the analysis from this frame.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--rdf",
        action="store_true",
        help="Plot the RDFs between tail group beads and exit.",
    )
    plot_options = parser.add_argument_group("Plot types")
    plot_options.add_argument(
        "--clustering",
        action="store_true",
        help="Compare the clustering behaviour of several simulations.",
    )
    plot_options.add_argument(
        "--agg-num",
        action="store_true",
        help="Compare the aggregation numbers in several simulations.",
    )
    plot_options.add_argument(
        "--rog",
        action="store_true",
        help="Compare the radius of gyration in several simulations.",
    )
    plot_options.add_argument(
        "--cpe",
        action="store_true",
        help="Compare the coordinate-pair eccentricities in several simulations.",
    )
    plot_options.add_argument(
        "--vol",
        action="store_true",
        help="Compare the aggregate volumes in several simulations.",
    )
    plot_options.add_argument(
        "--surf",
        action="store_true",
        help="Compare the surface areas in several simulations.",
    )
    plot_options.add_argument(
        "--sa-ratio",
        action="store_true",
        help="Compare the surface area to volume ratio in several simulations.",
    )
    plot_options.add_argument(
        "--hydrodynamic_radius",
        action="store_true",
        help="Compare the hydrodynamic radii in several simulations.",
    )
    args = parser.parse_args()

    WORKING_DIR = Path(args.dir)
    if not WORKING_DIR.exists():
        raise FileNotFoundError(f"Directory {WORKING_DIR} not found.")

    results_yaml = ResultsYAML(WORKING_DIR, args.r)
    results = results_yaml.get_results()

    if args.rdf:
        sns.set_theme(context="talk", style="darkgrid")
        tail_rdf(
            [result for result in results if isinstance(result, CoarseResults)],
            WORKING_DIR / "tail-rdf.pdf",
            step=args.step_size,
            start=args.start,
        )
        return

    properties = AggregateProperties.fast()
    if args.vol or args.surf or args.sa_ratio:
        properties |= {AggregateProperties.VOLUME, AggregateProperties.SURFACE_AREA}

    batch_ma_analysis(
        results,
        min_cluster_size=5,
        step=args.step_size,
        overwrite=args.overwrite,
        properties=properties,
    )

    if args.clustering:
        compare_clustering(results, WORKING_DIR / "clustering-comp.pdf")

    if args.agg_num:
        compare_dist(
            results,
            WORKING_DIR / "agg-num-comp.pdf",
            "Normalised aggregation numbers",
            ylim=(0, 1.01),
            hue=None,
        )

    if args.cpe:
        compare_cpe(results, WORKING_DIR / "cpe-comp.pdf")

    if args.rog:
        compare_val(
            results,
            WORKING_DIR / "rog-comp.pdf",
            "Radius of gyration",
        )

    if args.vol:
        compare_dist(
            results,
            WORKING_DIR / "vol-comp.pdf",
            "Volume",
            rename="Volume ($\\AA^3$)",
        )

    if args.surf:
        compare_dist(
            results,
            WORKING_DIR / "surf-comp.pdf",
            "Surface area",
            rename="Surface area ($\\AA^2$)",
        )
        compare_dist(
            results,
            WORKING_DIR / "norm-surf-comp.pdf",
            "Surfactant surface area",
            rename="Surface area per surfactant ($\\AA^2$)",
        )

    if args.sa_ratio:
        compare_dist(
            results,
            WORKING_DIR / "sa-ratio-comp.pdf",
            "Surface area / Volume",
            rename="Surface area / Volume ($\\AA^{-1}$)",
            ylim=(0, 1),
        )

    if args.hydrodynamic_radius:
        compare_dist(
            results,
            WORKING_DIR / "hydrodynamic-radius-comp.pdf",
            "Hydrodynamic radius",
            rename="Hydrodynamic radius ($\\AA$)",
        )


if __name__ == "__main__":
    main()
