"""Utilities for finding cluster sizes across simulations.

TODO:
    * Complete configuration of :class:`AggregateProperties` so that we can
      check whether a saved `DataFrame` has all the necessary columns.
    * Implement full CLI functionality and allow `SimResults` and
      `CoarseSimResults` to be stored in a JSON file.

"""

import argparse
import random
from collections import Counter, defaultdict
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, Optional, Union

import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import numpy as np
import pandas as pd
import pytim
import pyvista as pv
import seaborn as sns
from MDAnalysis.analysis.base import AnalysisBase, AtomGroup
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.core.groups import ResidueGroup
from pytim.datafiles import *
from scipy import spatial
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import euclidean
from tqdm import tqdm


def atom_to_mol_pairs(atom_pairs: np.ndarray, atom_per_mol: int) -> np.ndarray:
    """Convert an array of atom pairs to an array of molecule pairs."""
    # These are already sorted, so a floor division will give us the molecule idxs
    mol_pairs = (atom_pairs // atom_per_mol).astype(np.int32)
    mol_pairs = np.unique(mol_pairs, axis=0)

    # Get rid of self-connections
    mask = np.where(mol_pairs[:, 0] != mol_pairs[:, 1])

    return mol_pairs[mask]


def save_sparse(sparse_arrs: list[coo_array], file, compressed=True):
    """Save several sparse arrays to a file."""
    arrays_dict = {}
    for idx, mat in enumerate(sparse_arrs):
        arr_dict = {
            f"row{idx}": mat.row,
            f"col{idx}": mat.col,
            f"shape{idx}": mat.shape,
            f"data{idx}": mat.data,
        }
        arrays_dict.update(arr_dict)

    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)


def load_sparse(file) -> list[coo_array]:
    """Load a sparse array from disk."""
    sparse_arrs = []

    with np.load(file) as loaded:
        idx = 0
        while True:
            try:
                row = loaded[f"row{idx}"]
                col = loaded[f"col{idx}"]
                data = loaded[f"data{idx}"]
                shape = loaded[f"shape{idx}"]
            except KeyError as e:
                if idx == 0:
                    raise ValueError(
                        f"The file {file} does not contain any sparse matrices."
                    ) from e
                break

            sparse_arrs.append(coo_array((data, (row, col)), shape=shape))
            idx += 1

    return sparse_arrs


def asphericity(principal_moments: np.ndarray) -> float:
    """Calculate the asphericity from principal moments of the gyration tensor."""
    return principal_moments[2] - (principal_moments[0] + principal_moments[1]) / 2


def acylindricity(principal_moments: np.ndarray) -> float:
    """Calculate the acylindricity from principal moments of the gyration tensor."""
    return principal_moments[1] - principal_moments[0]


def anisotropy(principal_moments: np.ndarray) -> float:
    """Calculate the anisotropy from principal moments of the gyration tensor."""
    return (
        (3 / 2)
        * np.dot(principal_moments, principal_moments)
        / (np.sum(principal_moments) ** 2)
    ) - (1 / 2)


def get_conv(
    group: AtomGroup, radii_dict: dict = pytim_data.vdwradii(CHARMM27_TOP)
) -> tuple[float, list[float], list[float]]:
    u = group.universe

    # Get the surface
    wc = pytim.WillardChandler(
        u,
        group=group,
        alpha=3.0,
        mesh=1.1,
        fast=True,
        radii_dict=radii_dict,
        autoassign=False,
    )

    # converting PyTim to PyVista surface
    verts = wc.triangulated_surface[0]
    faces = wc.triangulated_surface[1]
    threes = 3 * np.ones((faces.shape[0], 1), dtype=int)
    faces = np.concatenate((threes, faces), axis=1)
    poly = pv.PolyData(verts, faces)

    # Get actual surface volume
    volume = poly.volume

    # Get convex hull's volume
    ch = spatial.ConvexHull(verts)
    ch_volume = ch.volume

    # Compute convexity
    convexity = volume / ch_volume

    # Get curvature data
    mean_curv = poly.curvature(curv_type="mean")
    G_curv = poly.curvature(curv_type="Gaussian")

    return convexity, mean_curv, G_curv


def get_surface(
    residues: ResidueGroup, radii_dict: dict = pytim_data.vdwradii(CHARMM27_TOP)
) -> tuple[float, int, dict[str, float]]:
    u = residues.universe

    # Get the surface
    surface = pytim.GITIM(
        u,
        group=residues.atoms,
        molecular=False,
        alpha=3.0,
        fast=True,
        radii_dict=radii_dict,
        autoassign=False,
    )

    surface_atoms = surface.layers[0]
    num_surface_atoms = len(surface_atoms)
    surface_ratio = num_surface_atoms / len(residues.atoms)

    num_surface_mols = len(np.unique(surface_atoms.resids))
    surface_counter = dict(Counter(surface_atoms.names))

    surface_types = {
        key: val / num_surface_atoms for key, val in surface_counter.items()
    }

    return surface_ratio, num_surface_mols, surface_types


def calc_semiaxis(atoms: AtomGroup):
    """
    Computes the semi-axes for a given selection of atoms in MD traj.

    Returns the semi-axis lengths as a numpy array from largest to smallest (a,
    b, c). Lengths have same units as MDAnalysis, default of Angstroms unless
    you specify otherwise

    """
    moments_val, princ_vec = np.linalg.eig(atoms.moment_of_inertia())
    micelle_mass = atoms.total_mass()

    # sortinng MoI and vectors by size of MoI
    idx = moments_val.argsort()[::-1]
    moments_val = moments_val[idx]
    princ_vec = princ_vec[:, idx]

    # Array to solve for axis lengths
    inverter = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * 0.5

    # converting MoI to axis length eigenvalues
    semiaxis = np.sqrt(
        (5 / micelle_mass) * (np.matmul(inverter, np.transpose(moments_val)))
    )

    # UNITS IN ANGSTROMS
    return np.flip(semiaxis)


def get_cpe(a, b, c):
    """
    Compute coordinate pair eccentricities for a given set of semi-axes.
    Returns 2 values, eab and eac, both on [0,1]

    a > b > c
    If you get errors, then your semi-axis order has been
    flipped somewhere.
    """
    eab = np.sqrt(1 - (b**2 / a**2))
    eac = np.sqrt(1 - (c**2 / a**2))
    return eab, eac


class AggregateProperties(Enum):
    AGGREGATION_NUMBERS = "Aggregation numbers"
    ASPHERICITIES = "Asphericities"
    ACYLINDRICITIES = "Acylindricities"
    ANISOTROPIES = "Anisotropies"
    CONVEXITIES = "Convexities"
    EAB = "Eab"
    EAC = "Eac"
    SURFACE_ATOM_RATIO = "Surface atom ratio"
    NUM_SURFACE_MOLECULES = "Num surface molecules"
    SURFACE_MOLECULE_RATIO = "Surface molecule ratio"
    SURFACE_ATOM_TYPES = "Surface atom types"
    MEAN_CURVATURES = "Mean curvatures"
    GAUSSIAN_CURVATURES = "Gaussian curvatures"

    @classmethod
    def all(cls) -> set["AggregateProperties"]:
        return set(cls)

    @classmethod
    def one_per_agg(cls):
        """Get the properties that are unique to each aggregate."""
        return {
            cls.AGGREGATION_NUMBERS,
            cls.ASPHERICITIES,
            cls.ACYLINDRICITIES,
            cls.ANISOTROPIES,
            cls.CONVEXITIES,
            cls.EAB,
            cls.EAC,
            cls.SURFACE_ATOM_RATIO,
            cls.NUM_SURFACE_MOLECULES,
            cls.SURFACE_MOLECULE_RATIO,
        }

    @classmethod
    def fast(cls):
        return cls.one_per_agg().difference(
            {
                cls.SURFACE_ATOM_RATIO,
                cls.SURFACE_MOLECULE_RATIO,
                cls.SURFACE_ATOM_TYPES,
                cls.CONVEXITIES,
                cls.NUM_SURFACE_MOLECULES,
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
    """Class for computing the adjacency matrix of surfactants in micelles."""

    NON_TYPE_COLS = [
        "Frame",
        "Time (ps)",
        "Aggregation numbers",
        "Asphericities",
        "Acylindricities",
        "Anisotropies",
        "Convexities",
        "Eab",
        "Eac",
        "Surface atom ratio",
        "Num surface molecules",
        "Surface molecule ratio",
        "Mean curvatures",
        "Gaussian curvatures",
    ]

    def __init__(
        self,
        tailgroups: AtomGroup,
        cutoff: float = 4.5,
        verbose=True,
        min_cluster_size: int = 5,
        properties: set[AggregateProperties] = AggregateProperties.all(),
        coarse: bool = False,
        **kwargs,
    ):
        """Split tails into different molecules."""
        trajectory = tailgroups.universe.trajectory
        super().__init__(trajectory, verbose, **kwargs)

        self.cutoff = cutoff
        self.tailgroups = tailgroups
        self.min_cluster_size = min_cluster_size
        self.properties = properties
        self.coarse = coarse

        self.num_surf: int = tailgroups.n_residues
        self.whole_molecules: ResidueGroup = tailgroups.residues
        self.atom_per_mol = int(len(self.tailgroups) / self.num_surf)

    @cached_property
    def vdwradii(self) -> dict[str, float]:
        """Determine the van der Waals radii for the atoms in the system."""
        if self.coarse:
            radii = dict()
            for atom in self.tailgroups.universe.atoms:
                # See Section 8.2 of
                # http://cgmartini.nl/index.php/martini-3-tutorials/parameterizing-a-new-small-molecule
                match atom.type[0]:
                    case "S":
                        radius = 0.225
                    case "T":
                        radius = 0.185
                    case _:
                        radius = 0.264

                radii[atom.name] = radius

            return radii

        else:
            return pytim_data.vdwradii(CHARMM36_TOP)

    def _prepare(self):
        """Initialise the results."""
        self.adj_mats: list[coo_array] = []

        self.frame_counter: list[int] = []
        self.time_counter: list[int] = []
        self.agg_nums: list[int] = []

        self.convexities: list[float] = []
        self.mean_curvs: list[list[float]] = []
        self.g_curvs: list[list[float]] = []

        # Coordinate pair eccentricities
        self.eabs: list[float] = []
        self.eacs: list[float] = []

        self.surface_atom_ratio: list[float] = []
        self.num_surface_mols: list[int] = []
        self.surface_mol_ratio: list[float] = []

        self.surface_types: list[dict[str, float]] = []

        self.asphericities: list[float] = []
        self.acylindricities: list[float] = []
        self.anisotropies: list[float] = []

    def _single_frame(self):
        """Calculate the contact matrix for the current frame."""
        atom_pairs = capped_distance(
            self.tailgroups,
            self.tailgroups,
            self.cutoff,
            box=self._ts.dimensions,
            return_distances=False,
        )
        mol_pairs = atom_to_mol_pairs(atom_pairs, self.atom_per_mol)
        num_pairs = mol_pairs.shape[0]
        ones = np.ones((num_pairs,), dtype=np.bool_)
        sparse_adj_arr = coo_array(
            (ones, (mol_pairs[:, 0], mol_pairs[:, 1])),
            shape=(self.num_surf, self.num_surf),
            dtype=np.bool_,
        )
        self.adj_mats.append(sparse_adj_arr)

        _, connected_comps = connected_components(sparse_adj_arr, directed=False)

        _, agg_indices, agg_nums = np.unique(
            connected_comps, return_counts=True, return_index=True
        )
        # agg_indices contains the index of the first molecule in each aggregate

        for i, agg_num in zip(range(len(agg_indices)), agg_nums):
            if agg_num < self.min_cluster_size:
                continue

            start_idx = agg_indices[i]
            try:
                end_idx = agg_indices[i + 1]
                agg_residues = self.whole_molecules[start_idx:end_idx]
            except IndexError:
                agg_residues = self.whole_molecules[start_idx:]

            princ_moms = agg_residues.gyration_moments()

            if self.properties.union(
                AggregateProperties.CONVEXITIES
                | AggregateProperties.MEAN_CURVATURES
                | AggregateProperties.GAUSSIAN_CURVATURES
            ):
                conv, mean_curv, G_curv = get_conv(agg_residues.atoms, self.vdwradii)
                self.convexities.append(conv)
                self.mean_curvs.append(mean_curv)
                self.g_curvs.append(G_curv)

            if self.properties.union(AggregateProperties.EAB | AggregateProperties.EAC):
                semiaxis = calc_semiaxis(agg_residues)
                cpe = get_cpe(*semiaxis)
                self.eabs.append(cpe[0])
                self.eacs.append(cpe[1])

            if self.properties.union(
                AggregateProperties.SURFACE_ATOM_RATIO
                | AggregateProperties.NUM_SURFACE_MOLECULES
                | AggregateProperties.SURFACE_MOLECULE_RATIO
                | AggregateProperties.SURFACE_ATOM_TYPES
            ):
                surface_ratio, num_surface_mols, surface_types = get_surface(
                    agg_residues
                )
                self.surface_atom_ratio.append(surface_ratio)
                self.num_surface_mols.append(num_surface_mols)
                self.surface_mol_ratio.append(num_surface_mols / agg_num)
                self.surface_types.append(surface_types)

            if AggregateProperties.ASPHERICITIES in self.properties:
                self.asphericities.append(asphericity(princ_moms))

            if AggregateProperties.ACYLINDRICITIES in self.properties:
                self.acylindricities.append(acylindricity(princ_moms))

            if AggregateProperties.ANISOTROPIES in self.properties:
                self.anisotropies.append(anisotropy(princ_moms))

            self.agg_nums.append(agg_num)

            self.frame_counter.append(self._ts.frame)
            self.time_counter.append(self._ts.time)

    def _conclude(self):
        """Store results in DataFrame."""
        self.df = pd.DataFrame(
            {
                "Frame": self.frame_counter,
                "Time (ps)": self.time_counter,
                "Aggregation numbers": self.agg_nums,
                "Asphericities": self.asphericities,
                "Acylindricities": self.acylindricities,
                "Anisotropies": self.anisotropies,
                "Convexities": self.convexities,
                "Eab": self.eabs,
                "Eac": self.eacs,
                "Surface atom ratio": self.surface_atom_ratio,
                "Num surface molecules": self.num_surface_mols,
                "Surface molecule ratio": self.surface_mol_ratio,
                "Surface atom types": self.surface_types,
                "Mean curvatures": self.mean_curvs,
                "Gaussian curvatures": self.g_curvs,
            }
        )

        surface_types_df = pd.DataFrame(self.surface_types)
        surface_types_df.fillna(0, inplace=True)

        self.df = pd.concat([self.df, surface_types_df], axis=1)

    def save(self, adj_path: Path, df_path: Path):
        save_sparse(self.adj_mats, adj_path)
        self.df.to_csv(df_path)


class Counterion(NamedTuple):
    shortname: str
    longname: str


CALCIUM = Counterion("ca", "Ca2+")
SODIUM = Counterion("na", "Na+")


class SimResults(NamedTuple):
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


class Coarseness(NamedTuple):
    dirname: str
    friendly_name: str


COARSEST = Coarseness("coarse-alpha", "Coarsest")
MIXED = Coarseness("mixed-alpha", "Mixed")
FINE = Coarseness("zeta", "Finest")


class CoarseSimResults(NamedTuple):
    """Information about some coarse-grained simulation results."""

    percent_aot: Union[int, float]
    coarseness: Coarseness
    tpr_file: Path
    traj_file: Path
    tail_match: str

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


class MCPosSolver:
    """Iteratively improve the arrangement of a multipartite graph using Monte Carlo."""

    def __init__(self, graph: nx.DiGraph, subset_key: str) -> None:
        self.graph = graph
        self.subset_key = subset_key

        self.starting_pos = nx.multipartite_layout(graph, subset_key, align="vertical")
        self.pos_array = np.vstack(list(self.starting_pos.values()))

        # Get a dict of node indexes that are in the same layer
        # as well as a list of each nodes' neighbours
        layers = defaultdict(list)
        self.node_neighbours: list[list[int]] = []
        for idx, (v, data) in enumerate(graph.nodes(data=True)):
            layers[data[subset_key]].append(idx)
            self.node_neighbours.append([int(nb) for nb in nx.all_neighbors(graph, v)])

        self.layers = dict(sorted(layers.items()))

        self.num_swaps: int = 0

    def try_swap(self, layer: int, idxs: tuple[int, int]) -> bool:
        """Try swapping two nodes in a layer."""
        curr_dist = 0
        first_node_idx = self.layers[layer][idxs[0]]
        second_node_idx = self.layers[layer][idxs[1]]

        for nb in self.node_neighbours[first_node_idx]:
            curr_dist += euclidean(
                self.pos_array[nb, :], self.pos_array[first_node_idx, :]
            )
        for nb in self.node_neighbours[second_node_idx]:
            curr_dist += euclidean(
                self.pos_array[nb, :], self.pos_array[second_node_idx, :]
            )

        new_dist = 0
        for nb in self.node_neighbours[first_node_idx]:
            new_dist += euclidean(
                self.pos_array[nb, :], self.pos_array[second_node_idx, :]
            )
        for nb in self.node_neighbours[second_node_idx]:
            new_dist += euclidean(
                self.pos_array[nb, :], self.pos_array[first_node_idx, :]
            )

        if new_dist < curr_dist:
            old_first_pos = self.pos_array[first_node_idx, :]
            self.pos_array[first_node_idx, :] = self.pos_array[second_node_idx, :]
            self.pos_array[second_node_idx, :] = old_first_pos
            return True
        else:
            return False

    def iterate(self, num_iterations: int = 100000):
        """Update the positions a set number of times."""
        for _ in tqdm(range(num_iterations), "Updating graph layout"):
            layer = random.sample(range(len(self.layers)), 1)[0]
            idxs = tuple(random.sample(range(len(self.layers[layer])), 2))

            if self.try_swap(layer, idxs):
                self.num_swaps += 1

    @property
    def pos_dict(self) -> dict:
        """Get the position dictionary."""
        return {key: val for key, val in zip(self.starting_pos.keys(), self.pos_array)}


def all_atomistic_ma(
    result: SimResults, min_cluster_size: int = 5, step: int = 1
) -> MicelleAdjacency:
    """Run a micelle adjacency analysis for the default tail group indices."""
    u = mda.Universe(result.tpr_file, result.traj_file)
    if step > 1:
        u.transfer_to_memory(step=step)

    # Find the atoms in tail groups
    tail_atom_nums = [6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20]
    tail_atom_names = [f"C{idx}" for idx in tail_atom_nums]
    sel_str = " or ".join([f"name {name}" for name in tail_atom_names])
    end_atoms = u.select_atoms(sel_str)

    ma = MicelleAdjacency(end_atoms, min_cluster_size=min_cluster_size)
    ma.run()

    return ma


def coarse_ma(
    result: CoarseSimResults, min_cluster_size: int = 5, step: int = 1
) -> MicelleAdjacency:
    """Run a micelle adjacency analysis for the default tail group indices."""
    u = mda.Universe(result.tpr_file, result.traj_file)
    if step > 1:
        u.transfer_to_memory(step=step)

    tail_atoms = u.select_atoms(result.tail_match)

    ma = MicelleAdjacency(tail_atoms, min_cluster_size=min_cluster_size, coarse=True)
    ma.run()

    return ma


def batch_ma_analysis(
    results: list[SimResults] | list[CoarseSimResults],
    min_cluster_size: int = 5,
    step: int = 1,
    only_last: bool = False,
    dir_: Path = Path("."),
) -> pd.DataFrame:
    """Load MA results from disk or run analyses anew."""
    plot_df = pd.DataFrame()
    for result in results:
        adj_path = dir_ / result.adj_file
        df_path = dir_ / result.df_file

        if df_path.exists():
            print(f"Found existing files for {result.plot_name}.")
            this_df = pd.read_csv(df_path, index_col=0)

        else:
            print(f"Analysing {result.plot_name} results.")

            if isinstance(result, CoarseSimResults):
                ma = coarse_ma(result, min_cluster_size, step)
            else:
                ma = all_atomistic_ma(result, min_cluster_size, step)
            ma.save(adj_path, df_path)

            this_df = ma.df

        if only_last:
            this_df = this_df[this_df["Frame"] == this_df["Frame"].max()]

        this_df = this_df[this_df["Aggregation numbers"] >= min_cluster_size]
        # TODO: Curvatures are lists of lists, so we need to do something about that
        # this_df = this_df.groupby("Time (ps)").mean()
        # this_df["Time (ps)"] = this_df.index
        this_df["% AOT"] = f"{result.percent_aot:.1f}"
        this_df["% AOT"] = this_df["% AOT"].astype("category")
        this_df["Simulation"] = result.plot_name
        if isinstance(result, SimResults):
            this_df["Counterion"] = result.counterion.longname

        plot_df = pd.concat([plot_df, this_df], ignore_index=True)

    return plot_df


def plot_agg_events(result: SimResults, dir_: Path = Path(".")):
    """Plot aggregation events."""
    # * Load adjacency matrix
    adj_path = dir_ / result.adj_file
    adj_mats = load_sparse(adj_path)

    agg_adj_path = dir_ / result.agg_adj_file
    try:
        G = nx.read_gml(agg_adj_path)
    except FileNotFoundError:
        # We're going to make one big graph. Each node represents an aggregate at a
        # given timestep. Edges represent aggregation events, in which at least one
        # molecule from an aggregate becomes part of another aggregate.
        G = nx.DiGraph()
        counter: int = 0
        this_adj_mat = adj_mats[0]

        n_connected, these_con_labels = connected_components(
            this_adj_mat, directed=False, return_labels=True
        )
        these_nodes: np.ndarray = (
            these_con_labels + counter
        )  # The node index that each molecule belongs to

        these_unique, these_sizes = np.unique(these_nodes, return_counts=True)
        G.add_nodes_from(
            [
                (idx, {"size": size, "frame": 0})
                for idx, size in np.c_[these_unique, these_sizes]
            ]
        )

        for i in tqdm(range(1, len(adj_mats)), "Finding connectivity changes"):
            counter += n_connected
            next_adj_mat = adj_mats[i]

            n_connected, next_con_labels = connected_components(
                next_adj_mat, directed=False, return_labels=True
            )
            next_nodes: np.ndarray = next_con_labels + counter  # The new node index

            next_unique, next_sizes = np.unique(next_nodes, return_counts=True)
            # Add new nodes
            G.add_nodes_from(
                [
                    (idx, {"size": size, "frame": i})
                    for idx, size in np.c_[next_unique, next_sizes]
                ]
            )
            # Add edges between previous and new nodes
            G.add_edges_from(np.c_[these_nodes, next_nodes])

            these_nodes = next_nodes

        nx.write_gml(G, agg_adj_path, str)

    # * Plot the directed graph
    # node_sizes = [int(data["size"]) for _, data in G.nodes(data=True)]
    pos_solver = MCPosSolver(G, "frame")
    # pos_solver.iterate()

    # print(f"{pos_solver.num_swaps=}")

    pos = pos_solver.pos_dict
    node_sizes = [int(data["size"]) / 10 for _, data in G.nodes(data=True)]
    plt.figure(figsize=(30, 5))

    cmap = "viridis"
    vmin = min(node_sizes)
    vmax = max(node_sizes)

    nx.draw(
        G,
        pos=pos,
        arrows=False,
        node_size=node_sizes,
        node_color=node_sizes,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        width=0.1,
        edge_color="gray",
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin * 10, vmax=vmax * 10)
    )
    sm._A = []

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(sm, ax=ax)

    plt.savefig(f"{result.name}-agg-events.png", bbox_inches="tight", dpi=500)


def compare_val(
    results: list[SimResults] | list[CoarseSimResults],
    graph_file: Path,
    y_axis: str,
    min_cluster_size: int = 5,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = batch_ma_analysis(results, min_cluster_size)

    TIME_COL = "Time (ns)"
    plot_df[TIME_COL] = plot_df["Time (ps)"] * 1e-3
    plot_df.drop(columns=["Frame", "Time (ps)"], inplace=True)

    print("Done analysing results!")
    print("Plotting graphs.")

    g = sns.relplot(
        data=plot_df,
        x=TIME_COL,
        y=y_axis,
        col="% AOT",
        row="Counterion",
        hue="Counterion",
        kind="line",
        errorbar="sd",
        facet_kws={"margin_titles": True, "despine": False, "sharey": "row"},
    )
    g.tight_layout()
    g.savefig(graph_file, transparent=True)


def compare_dist(
    results: list[SimResults] | list[CoarseSimResults],
    graph_file: Path,
    y_axis: str,
    use_interval: bool = True,
    interval: float = 20.0,
    min_cluster_size: int = 5,
    ylim: Optional[tuple[float, float]] = None,
    semilog: bool = False,
    use_hue: bool = True,
    rename: Optional[str] = None,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = batch_ma_analysis(results, min_cluster_size)

    TIME_COL = "Time (ns)"
    plot_df[TIME_COL] = plot_df["Time (ps)"] * 1e-3
    plot_df.drop(columns=["Frame", "Time (ps)"], inplace=True)

    if use_interval:
        plot_df = plot_df[plot_df[TIME_COL] % interval == 0]

    plot_df[TIME_COL] = plot_df[TIME_COL].apply(lambda x: f"{x:.0f}")
    time_labels = sorted(plot_df[TIME_COL].unique(), key=lambda x: int(x))

    plot_df["Log agg. num"] = plot_df["Aggregation numbers"].apply(np.log10)

    if rename is not None:
        plot_df[rename] = plot_df[y_axis]

    print("Done analysing results!")
    print("Plotting graphs.")

    g = sns.catplot(
        data=plot_df,
        x=TIME_COL,
        order=time_labels,
        y=y_axis if not rename else rename,
        col="% AOT",
        row="Counterion",
        hue="Log agg. num" if use_hue else None,
        kind="strip",
        # inner=None,
        sharey=True,
        facet_kws={"margin_titles": True, "despine": False},
        palette="flare",
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

    g.set_xticklabels(time_labels, rotation=45)
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def compare_final_types(
    results: list[SimResults] | list[CoarseSimResults],
    graph_file: Path,
    min_cluster_size: int = 5,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = batch_ma_analysis(results, min_cluster_size, only_last=True)
    plot_df.drop(columns=MicelleAdjacency.NON_TYPE_COLS, inplace=True)

    plot_df = plot_df.groupby(["Counterion", "% AOT"]).sum().reset_index()

    print("Done analysing results!")
    print("Plotting graphs.")

    g = sns.catplot(
        data=plot_df,
        kind="bar",
        col="% AOT",
        row="Counterion",
        sharey=True,
        facet_kws={"margin_titles": True, "despine": False},
    )

    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def compare_cpe(
    results: list[SimResults] | list[CoarseSimResults],
    graph_file: Path,
    use_interval: bool = True,
    interval: float = 20.0,
    min_cluster_size: int = 5,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = batch_ma_analysis(results, min_cluster_size)

    TIME_COL = "Time (ns)"
    plot_df[TIME_COL] = plot_df["Time (ps)"] * 1e-3
    plot_df.drop(columns=["Frame", "Time (ps)"], inplace=True)

    if use_interval:
        plot_df = plot_df[plot_df[TIME_COL] % interval == 0]

    plot_df["Log agg. num"] = plot_df["Aggregation numbers"].apply(np.log10)

    EAB = "$e_{ab}$"
    EAC = "$e_{ac}$"

    plot_df["$e_{ab}$"] = plot_df["Eab"]
    plot_df["$e_{ac}$"] = plot_df["Eac"]

    print("Done analysing results!")
    print("Plotting graphs.")

    g = sns.relplot(
        kind="scatter",
        data=plot_df,
        x=EAB,
        y=EAC,
        col="% AOT",
        row="Counterion",
        hue="Log agg. num",
        facet_kws={"margin_titles": True, "despine": False},
        palette="flare",
    )
    g.set(xlim=(0, 1), ylim=(0, 1))
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def compare_clustering(
    results: list[SimResults] | list[CoarseSimResults],
    graph_file: Path,
    min_cluster_size: int = 5,
    dir_: Path = Path("."),
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = pd.DataFrame()
    for result in results:
        adj_path = dir_ / result.adj_file
        df_path = dir_ / result.df_file

        if df_path.exists():
            print(f"Found existing files for {result.plot_name}.")
            this_df = pd.read_csv(df_path, index_col=0)

        else:
            print(f"Analysing {result.plot_name} results.")

            ma = default_tail_ma(result)
            ma.save(adj_path, df_path)

            this_df = ma.df

        this_df = this_df[this_df["Aggregation numbers"] >= min_cluster_size]
        this_df = this_df.groupby("Time (ps)").mean()
        this_df["Time (ps)"] = this_df.index
        this_df["% AOT"] = result.percent_aot
        this_df["Simulation"] = result.plot_name

        plot_df = pd.concat([plot_df, this_df], ignore_index=True)

    TIME_COL = "Time (ns)"
    plot_df[TIME_COL] = plot_df["Time (ps)"] * 1e-3
    plot_df.drop(columns=["Frame", "Time (ps)"], inplace=True)

    print("Done analysing results!")
    print("Plotting graphs.")

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
        facet_kws={"margin_titles": True, "despine": False, "sharey": "row"},
    )

    g.fig.suptitle(f"Clustering comparison with min cluster size = {min_cluster_size}")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels(TIME_COL, "Value")
    g.tight_layout()
    g.savefig(graph_file)


if __name__ == "__main__":
    sns.set_theme(context="talk", style="dark", palette="flare")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory to look in for files.",
    )
    args = parser.parse_args()

    # all_results = [
    #     SimResults(
    #         1,
    #         SODIUM,
    #         HERE / "1-na-no-water.tpr",
    #         HERE / "1-na-no-water.xtc",
    #     ),
    #     SimResults(
    #         1,
    #         CALCIUM,
    #         HERE / "1-ca-no-water.tpr",
    #         HERE / "1-ca-no-water.xtc",
    #     ),
    #     SimResults(
    #         7.2,
    #         SODIUM,
    #         HERE / "7_2-na-no-water.tpr",
    #         HERE / "7_2-na-no-water.xtc",
    #     ),
    #     SimResults(
    #         7.2,
    #         CALCIUM,
    #         HERE / "7_2-ca-no-water.tpr",
    #         HERE / "7_2-ca-no-water.xtc",
    #     ),
    #     SimResults(
    #         20,
    #         SODIUM,
    #         HERE / "20-na-no-water.tpr",
    #         HERE / "20-na-no-water.xtc",
    #     ),
    #     SimResults(
    #         20,
    #         CALCIUM,
    #         HERE / "20-ca-no-water.tpr",
    #         HERE / "20-ca-no-water.xtc",
    #     ),
    # ]

    # get_agg_nums(all_results, min_cluster_size=5, step=100)

    # compare_clustering(all_results, HERE / "clustering-comp.png")
    # compare_val(all_results, HERE / "num-clusters-comp-new.png", "Num clusters")
    # compare_val(all_results, HERE / "agg-num-comp-new.png", "Aggregation numbers")
    # compare_val(all_results, HERE / "asp-num-comp.png", "Asphericities")
    # compare_val(all_results, HERE / "aniso-num-comp.png", "Anisotropies")
    # compare_dist(all_results, HERE / "aniso-dist-comp.png", "Anisotropies")
    # compare_val(all_results, HERE / "acy-num-comp.png", "Acylindricities")
    # compare_dist(
    #     all_results, HERE / "acy-dist-comp.png", "Acylindricities", semilog=True
    # )
    # compare_dist(
    #     all_results,
    #     HERE / "agg-num-dist-comp.pdf",
    #     "Aggregation numbers",
    #     use_hue=False,
    #     semilog=True,
    # )
    # compare_dist(all_results, HERE / "asp-dist-comp.png", "Asphericities", semilog=True)

    # compare_dist(all_results, HERE / "mean-curv-dist-comp.png", "Mean curvatures")
    # compare_dist(all_results, HERE / "g-curv-dist-comp.png", "Gaussian curvatures")
    # compare_dist(all_results, HERE / "convex-dist-comp.pdf", "Convexities", ylim=(0, 1))
    # compare_cpe(all_results, HERE / "cpe-comp.pdf")
    # compare_dist(all_results, HERE / "eab-dist-comp.pdf", "Eab", rename="$e_{ab}$")

    # compare_dist(
    #     all_results, HERE / "surf-atom-dist-comp.pdf", "Surface atom ratio", ylim=(0, 1)
    # )
    # compare_dist(
    #     all_results,
    #     HERE / "surf-mol-dist-comp.pdf",
    #     "Surface molecule ratio",
    #     ylim=(0, 1),
    # )
    # compare_final_types(all_results, HERE / "final-types-comp.pdf")

    # for result in all_results:
    #     plot_agg_events(result)
