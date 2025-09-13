"""Utilities for finding cluster sizes across simulations.

TODO:
    * Complete configuration of :class:`AggregateProperties` so that we can
      check whether a saved `DataFrame` has all the necessary columns.
    * Implement full CLI functionality and allow `SimResults` and
      `CoarseSimResults` to be stored in a JSON file.

"""

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from pathlib import Path
from typing import Optional

from MDAnalysis.analysis.results import ResultsGroup

try:
    import ase
    from dscribe.descriptors import SOAP
    from sklearn.decomposition import KernelPCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    DSCRIBE_IMPORT_ERROR = None
    HAS_DSCRIBE = True
except ImportError as e:
    DSCRIBE_IMPORT_ERROR = e
    HAS_DSCRIBE = False

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytim
import pyvista as pv

try:
    from kdecv.calculate import GaussianKDE, periodic_stddev

    HAS_KDECV: bool = True
except ImportError:
    HAS_KDECV = False
from MDAnalysis.analysis.base import AnalysisBase, AtomGroup
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.core.groups import ResidueGroup
from pytim.datafiles import CHARMM27_TOP, pytim_data
from pytim.interface import Interface

try:
    from scipy.sparse import coo_array
except ImportError:
    from scipy.sparse import coo_matrix as coo_array

from scipy.sparse.csgraph import connected_components

from .utilities import *


def check_dscribe():
    """Check if `dscribe` is installed."""
    if not HAS_DSCRIBE:
        raise ImportError(
            "Cannot load dependencies to calculate SOAP KPCA"
        ) from DSCRIBE_IMPORT_ERROR


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


def vesicality(tailgroups: AtomGroup):
    r"""Determine 'vesicality' of a cluster.

    'Vesicality' is a measure of how well the aggregate matches a spherical
    bilayer. We first describe the molecules in the cluster as vectors, pointing
    from the centre of the tail group to the centre of the head group. We
    express this vector field in spherical coordinates. In a vesicle, there will
    be a pair of antisymmetric vectors at a similar theta/phi. We can take the
    dot product of the orientation vector with its displacement vector from the
    center of geometry of the cluster to get the scalar orientation relative to
    the centre, $o$.

    We can partition the aggregate into two layers: inward ($o < 0$) and outward
    ($o > 0$). The expected area of a shell is $A = 4 \pi r^2$, where $r$ is the
    radius of the shell, and the expected number of surfactant molecules should
    be proportional. Therefore, the ratio of the number in each layer is given
    by $N_{in} / N_{out} = r_{in}^2 / r_{out}^2$. We can therefore calculate the
    vesicality as $v = \frac{N_{in} r_{out}^2}{N_{out} r_{in}^2}$. If $N_{in} =
    0$, we define $v = 0$.

    TODO: This function doesn't quite work as described. We use the sum of the
    $o$ values instead of $N$, and the weighted averages of the $r$ values. This
    was done to avoid issues where curved, but not closed bicelles would have
    high vesicality values. This also means that the interpretation of the
    metric needs to change. Generally, values close to, or greater than 1 are
    vesicles.

    """
    agg_cog = tailgroups.center_of_geometry()

    outer_hg_radii: list[float] = []
    inner_hg_radii: list[float] = []
    outer_weights: list[float] = []
    inner_weights: list[float] = []

    resids = tailgroups.residues.unique

    for res in resids:
        headgroup = res.atoms.difference(tailgroups)
        tailgroup = res.atoms.intersection(tailgroups)

        hg_cog = headgroup.center_of_geometry()
        tail_cog = tailgroup.center_of_geometry()

        # Get the orientation vector
        orientation = hg_cog - tail_cog
        orientation /= np.linalg.norm(orientation)

        from_agg_centre = hg_cog - agg_cog
        dist_from_centre = np.linalg.norm(from_agg_centre)

        from_agg_centre /= dist_from_centre
        scalar_orientation = np.dot(orientation, from_agg_centre)

        radii_list = outer_hg_radii if scalar_orientation > 0 else inner_hg_radii
        weights_list = outer_weights if scalar_orientation > 0 else inner_weights

        radii_list.append(dist_from_centre)
        weights_list.append(np.abs(scalar_orientation))

    # Get the number of molecules in each layer
    n_outer = np.sum(outer_weights)
    n_inner = np.sum(inner_weights)

    if n_inner == 0:
        return 0
    if n_outer == 0:
        print("No outer layer headgroups found -- could be an inverse micelle?")
        return 0

    # Get the average radius of each layer
    r_outer = np.average(outer_hg_radii, weights=outer_weights)
    r_inner = np.average(inner_hg_radii, weights=inner_weights)

    # Calculate the vesicality
    return min((n_inner * r_outer**2) / (n_outer * r_inner**2), 1.0)


def radius_of_gyration(group: AtomGroup) -> float:
    """Calculate the radius of gyration for a given group of atoms."""
    return group.radius_of_gyration()


def willard_chandler(
    group: AtomGroup,
    radii_dict: dict = pytim_data.vdwradii(CHARMM27_TOP),
    write: Optional[str] = None,
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
        # centered=True,
    )

    if write is not None:
        wc.writecube(write, group=group)
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


def count_inside_vesicle(
    aggregate: AtomGroup, tailgroups: AtomGroup, resname: str
) -> int:
    """Count the number of entities with a specific resname inside the vesicle.

    The vesicle interior is defined as a sphere with radius equal to the average
    of the inner and outer headgroup layer radii.
    """
    agg_cog = tailgroups.center_of_geometry()
    universe = tailgroups.universe

    # Calculate inner and outer radii similar to vesicality()
    outer_hg_radii: list[float] = []
    inner_hg_radii: list[float] = []
    outer_weights: list[float] = []
    inner_weights: list[float] = []

    resids = tailgroups.residues.unique

    for res in resids:
        headgroup = res.atoms.difference(tailgroups)
        tailgroup = res.atoms.intersection(tailgroups)

        hg_cog = headgroup.center_of_geometry()
        tail_cog = tailgroup.center_of_geometry()

        # Get the orientation vector
        orientation = hg_cog - tail_cog
        orientation /= np.linalg.norm(orientation)

        from_agg_centre = hg_cog - agg_cog
        dist_from_centre = np.linalg.norm(from_agg_centre)

        from_agg_centre /= dist_from_centre
        scalar_orientation = np.dot(orientation, from_agg_centre)

        radii_list = outer_hg_radii if scalar_orientation > 0 else inner_hg_radii
        weights_list = outer_weights if scalar_orientation > 0 else inner_weights

        radii_list.append(dist_from_centre)
        weights_list.append(np.abs(scalar_orientation))

    # If there's no inner layer, return 0
    if not inner_hg_radii or not outer_hg_radii:
        return 0

    # Get the average radius of each layer
    r_outer = np.average(outer_hg_radii, weights=outer_weights)
    r_inner = np.average(inner_hg_radii, weights=inner_weights)

    # Calculate shell radius (average of inner and outer radii)
    shell_radius = r_inner  # Use inner radius as cutoff

    # Select entities with matching resname
    target_entities = aggregate.universe.select_atoms(f"resname {resname}")

    # Count entities within the shell radius from the center of geometry
    count = 0
    box = aggregate.universe.dimensions
    for atom in target_entities:
        # Calculate distance considering PBC
        diff = atom.position - agg_cog
        # Apply minimum image convention
        for i in range(3):
            if diff[i] > box[i] / 2:
                diff[i] -= box[i]
            elif diff[i] < -box[i] / 2:
                diff[i] += box[i]
        distance = np.linalg.norm(diff)
        if distance < shell_radius:
            count += 1

    return count


def count_inner_aot(tailgroups: AtomGroup) -> int:
    """Count the number of AOT molecules in the inner layer of a vesicle."""
    agg_cog = tailgroups.center_of_geometry()
    inner_layer_count = 0

    resids = tailgroups.residues.unique

    for res in resids:
        headgroup = res.atoms.difference(tailgroups)
        tailgroup = res.atoms.intersection(tailgroups)

        hg_cog = headgroup.center_of_geometry()
        tail_cog = tailgroup.center_of_geometry()

        # Get the orientation vector
        orientation = hg_cog - tail_cog
        orientation /= np.linalg.norm(orientation)

        from_agg_centre = hg_cog - agg_cog
        dist_from_centre = np.linalg.norm(from_agg_centre)

        from_agg_centre /= dist_from_centre
        scalar_orientation = np.dot(orientation, from_agg_centre)

        # If scalar_orientation is negative, it's an inner layer molecule
        if scalar_orientation < 0:
            inner_layer_count += 1

    return inner_layer_count


def get_cpe(atoms: AtomGroup):
    """
    Compute coordinate pair eccentricities for a given set of semi-axes.
    Returns 2 values, eab and eac, both on [0,1]
    """
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


def atoms_to_ase(
    atoms: mda.AtomGroup, atom_map: Optional[dict[str, int]] = None
) -> "ase.Atoms":
    """Convert MDAnalysis atoms to an ASE Atoms object."""
    check_dscribe()

    u = atoms.universe

    symbols = [atom.type[0] for atom in atoms]
    if atom_map is not None:
        numbers = [atom_map[symbol] for symbol in symbols]
    else:
        numbers = ase.data.atomic_numbers[symbols]

    return ase.Atoms(
        numbers=numbers,
        positions=atoms.positions,
        masses=atoms.masses,
        cell=u.dimensions[:3],
        pbc=True,
    )


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

        self.results = ClusteringResults()

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")

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

    @cached_property
    def atom_map(self) -> dict[str, int]:
        """Get a canonical map from atom types to integers."""
        atom_types = set(atom.type[0] for atom in self.whole_molecules.atoms)
        return {atom: idx + 1 for idx, atom in enumerate(sorted(atom_types))}

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
        """Initialise the soap parameteriser."""
        self.soap: Optional[SOAP] = None
        if HAS_DSCRIBE:
            r_cut = 3 * self.cutoff

            self.soap = SOAP(
                species=self.atom_map.values(),
                r_cut=r_cut,
                n_max=8,
                l_max=8,
                sparse=False,
                periodic=True,
                # weighting={"function": "poly", "r0": r_cut, "m": 1, "c": 1, "w0": 0},
                average="inner",
            )

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

            center_on_cluster(agg_residues.atoms)

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
                vol, surf = willard_chandler(
                    agg_residues.atoms,
                    self.vdwradii,
                )
                if current_idx is None:
                    self.results.volume.append(vol)
                    self.results.surface.append(surf)
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
                    self.results.eabs.append(eab)
                    self.results.eacs.append(eac)
                else:
                    self.df.loc[current_idx, AggregateProperties.EAB.value] = eab
                    self.df.loc[current_idx, AggregateProperties.EAC.value] = eac

            if self.do_calculate(
                AggregateProperties.RADIUS_OF_GYRATION, current_agg_entry
            ):
                if current_idx is None:
                    self.results.radii_of_gyration.append(
                        radius_of_gyration(agg_residues.atoms)
                    )
                else:
                    self.df.loc[
                        current_idx, AggregateProperties.RADIUS_OF_GYRATION.value
                    ] = radius_of_gyration(agg_residues.atoms)

            if self.do_calculate(AggregateProperties.TOTAL_VOLUME, current_agg_entry):
                if not HAS_KDECV:
                    raise ImportError(
                        "Cannot load library `kdecv`, required for total volume calculation. Consider using `--vol` instead."
                    )

                # Negatively weight the headgroups
                weights = np.ones(len(agg_residues.atoms))
                headgroup_idxs = np.where(
                    [not atom in self.tailgroups for atom in agg_residues.atoms]
                )
                weights[headgroup_idxs] *= -1

                box_dim = self._ts.dimensions[:3]

                gaus_kde = GaussianKDE(
                    agg_residues.center_of_geometry(compound="residues"),
                    box_dim,
                    bw_method=3.0
                    / periodic_stddev(self.whole_molecules.atoms.positions, box_dim),
                    # weights=weights,
                )

                total_vol = gaus_kde.volume_estimate(
                    2.0,
                    rel_threshold=1 / 3,
                    smooth_cutoff=1e-6,
                )

                if current_idx is None:
                    self.results.total_volume.append(total_vol)
                else:
                    self.df.loc[current_idx, AggregateProperties.TOTAL_VOLUME.value] = (
                        total_vol
                    )

            if self.do_calculate(
                AggregateProperties.AGGREGATION_NUMBERS, current_agg_entry
            ):
                if current_idx is None:
                    self.results.agg_nums.append(agg_num)
                else:
                    self.df.loc[
                        current_idx, AggregateProperties.AGGREGATION_NUMBERS.value
                    ] = agg_num

            if self.do_calculate(
                AggregateProperties.NORMALISED_AGGREGATION_NUMBERS, current_agg_entry
            ):
                norm_agg_num = agg_num / self.num_surf
                if current_idx is None:
                    self.results.norm_agg_nums.append(norm_agg_num)
                else:
                    self.df.loc[
                        current_idx,
                        AggregateProperties.NORMALISED_AGGREGATION_NUMBERS.value,
                    ] = norm_agg_num

            if self.do_calculate(AggregateProperties.VESICALITY, current_agg_entry):
                vesicality_value = vesicality(agg_residues.atoms & self.tailgroups)
                if current_idx is None:
                    self.results.vesicalities.append(vesicality_value)
                else:
                    self.df.loc[current_idx, AggregateProperties.VESICALITY.value] = (
                        vesicality_value
                    )

            if self.do_calculate(
                AggregateProperties.COUNTERIONS_INSIDE, current_agg_entry
            ):
                counterions_count = count_inside_vesicle(
                    agg_residues, agg_residues.atoms & self.tailgroups, "NA"
                )
                if current_idx is None:
                    self.results.counterions_inside.append(counterions_count)
                else:
                    self.df.loc[
                        current_idx, AggregateProperties.COUNTERIONS_INSIDE.value
                    ] = counterions_count

            if self.do_calculate(AggregateProperties.WATER_INSIDE, current_agg_entry):
                water_count = count_inside_vesicle(
                    agg_residues, agg_residues.atoms & self.tailgroups, "W"
                )
                if current_idx is None:
                    self.results.water_inside.append(water_count)
                else:
                    self.df.loc[current_idx, AggregateProperties.WATER_INSIDE.value] = (
                        water_count
                    )

            if self.do_calculate(AggregateProperties.INNER_AOT, current_agg_entry):
                inner_aot_count = count_inner_aot(agg_residues.atoms & self.tailgroups)
                if current_idx is None:
                    self.results.inner_aot.append(inner_aot_count)
                else:
                    self.df.loc[current_idx, AggregateProperties.INNER_AOT.value] = (
                        inner_aot_count
                    )

            if self.do_calculate(
                AggregateProperties.SOAP_SIM_1 | AggregateProperties.SOAP_SIM_2,
                current_agg_entry,
            ):
                check_dscribe()

                # Convert the atoms to an ASE Atoms object
                ase_atoms = atoms_to_ase(agg_residues.atoms, atom_map=self.atom_map)

                # Get the average SOAP vector
                soap_vector = self.soap.create(ase_atoms, n_jobs=-1)
                if current_idx is None:
                    self.results.soap_vectors.append(soap_vector)
                else:
                    self.df.loc[current_idx, AggregateProperties.SOAP_VECTOR.value] = (
                        soap_vector.tostring()
                    )

            if current_idx is None:
                self.results.frame_counter.append(self._ts.frame)
                self.results.time_counter.append(self._ts.time)

            # TODO: This is why the min_cluster_size must be the same when updating.
            # TODO: This index doesn't keep track of how many aggregates below the threshold were skipped
            agg_idx += 1

    def _get_aggregator(self) -> ResultsGroup:
        return ResultsGroup(
            lookup={key: ResultsGroup.flatten_sequence for key in self.results.computed}
        )

    def _conclude(self):
        """Store results in DataFrame and calculate SOAP KPCA."""

        data = {
            "Frame": self.results.frame_counter,
            "Time (ps)": self.results.time_counter,
            AggregateProperties.AGGREGATION_NUMBERS.value: self.results.agg_nums,
            AggregateProperties.NORMALISED_AGGREGATION_NUMBERS.value: self.results.norm_agg_nums,
            AggregateProperties.EAB.value: self.results.eabs,
            AggregateProperties.EAC.value: self.results.eacs,
            AggregateProperties.RADIUS_OF_GYRATION.value: self.results.radii_of_gyration,
            AggregateProperties.VOLUME.value: self.results.volume,
            AggregateProperties.SURFACE_AREA.value: self.results.surface,
            AggregateProperties.TOTAL_VOLUME.value: self.results.total_volume,
            AggregateProperties.VESICALITY.value: self.results.vesicalities,
            AggregateProperties.COUNTERIONS_INSIDE.value: self.results.counterions_inside,
            AggregateProperties.WATER_INSIDE.value: self.results.water_inside,
            AggregateProperties.INNER_AOT.value: self.results.inner_aot,
            AggregateProperties.SOAP_VECTOR.value: self.results.soap_vectors,
        }
        data = {key: val for key, val in data.items() if len(val)}

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)
        self.df.sort_values("Frame", inplace=True, ignore_index=True)

        if (
            AggregateProperties.SOAP_VECTOR.value in self.df.columns
            and self.properties.intersection(
                {AggregateProperties.SOAP_SIM_1, AggregateProperties.SOAP_SIM_2}
            )
        ):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "kpca",
                        KernelPCA(n_components=2, kernel="poly", degree=2, coef0=0),
                    ),
                ]
            )

            soap_vectors = [
                np.fromstring(x) for x in self.df[AggregateProperties.SOAP_VECTOR.value]
            ]

            soap_sim = pipe.fit_transform(soap_vectors)
            soap_sim_1, soap_sim_2 = soap_sim[:, 0], soap_sim[:, 1]

            self.df[AggregateProperties.SOAP_SIM_1.value] = soap_sim_1
            self.df[AggregateProperties.SOAP_SIM_2.value] = soap_sim_2

        # Don't save the SOAP vectors to disk. This bloats the CSV unnecessarily.
        self.df.drop(
            columns=[AggregateProperties.SOAP_VECTOR.value],
            errors="ignore",
            inplace=True,
        )

    def save(self, adj_path: Path, df_path: Path):
        save_sparse(self.adj_mats, adj_path)
        self.df.to_csv(df_path)


def all_atomistic_ma(
    result: AtomisticResults,
    min_cluster_size: int = 5,
    step: int = 1,
    properties: "set[AggregateProperties]" = AggregateProperties.fast(),
    current_df: Optional[pd.DataFrame] = None,
    current_adj_mats: Optional[dict[int, coo_array]] = None,
    end: Optional[int] = None,
    num_workers: int = 1,
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
    ma.run(
        step=step,
        stop=end,
        n_workers=num_workers,
        backend="dask" if num_workers > 1 else "serial",
    )

    return ma


def coarse_ma(
    result: CoarseResults,
    min_cluster_size: int = 5,
    step: int = 1,
    properties: "set[AggregateProperties]" = AggregateProperties.fast(),
    current_df: Optional[pd.DataFrame] = None,
    current_adj_mats: Optional[dict[int, coo_array]] = None,
    end: Optional[int] = None,
    num_workers: int = 1,
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
    ma.run(
        step=step,
        stop=end,
        n_workers=num_workers,
        backend="dask" if num_workers > 1 else "serial",
    )

    return ma


def batch_ma_analysis(
    results: "list[AtomisticResults | CoarseResults]",
    min_cluster_size: int = 5,
    step: int = 1,
    only_last: bool = False,
    dir_: Path = Path("."),
    overwrite: bool = False,
    properties: "set[AggregateProperties]" = AggregateProperties.fast(),
    end: Optional[int] = None,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Load MA results from disk or run analyses anew."""
    plot_df = pd.DataFrame()
    for result in results:
        adj_path = dir_ / result.adj_file
        df_path = dir_ / result.df_file

        this_df = None
        current_adj_mats = None

        if df_path.exists():
            print(f"Found existing DataFrame for {result.plot_name}.")
            this_df = pd.read_csv(df_path, index_col=0)
        if adj_path.exists():
            print(f"Found existing adjacency matrix for {result.plot_name}.")
            current_adj_mats = load_sparse(adj_path)
        if overwrite:
            this_df.drop(
                columns=[prop.value for prop in properties],
                errors="ignore",
                inplace=True,
            )

        print(f"Analysing {result.plot_name} results.")

        if isinstance(result, CoarseResults):
            ma = coarse_ma(
                result,
                min_cluster_size,
                step,
                properties=properties,
                current_df=this_df,
                current_adj_mats=current_adj_mats,
                end=end,
                num_workers=num_workers,
            )
        else:
            ma = all_atomistic_ma(
                result,
                min_cluster_size,
                step,
                properties=properties,
                current_df=this_df,
                current_adj_mats=current_adj_mats,
                end=end,
                num_workers=num_workers,
            )
        ma.save(adj_path, df_path)

        this_df = ma.df.copy()

        if only_last:
            this_df = this_df.loc[this_df["Frame"] == this_df["Frame"].max()]

        this_df = this_df.loc[
            this_df[AggregateProperties.AGGREGATION_NUMBERS.value] >= min_cluster_size
        ]

        this_df["% AOT"] = f"{result.percent_aot:.1f}"

        this_df["% AOT"] = this_df["% AOT"].astype("category")
        this_df["Simulation"] = result.plot_name
        if isinstance(result, AtomisticResults):
            this_df["Type"] = result.counterion.longname
        else:
            this_df["Type"] = result.coarseness.friendly_name

        plot_df = pd.concat([plot_df, this_df], ignore_index=True)

    return plot_df
