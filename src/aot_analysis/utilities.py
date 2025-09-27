from concurrent.futures import ThreadPoolExecutor as Pool
from enum import Enum
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional, Union

import MDAnalysis as mda
import numpy as np
import yaml
from MDAnalysis.analysis.results import Results
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

try:
    from scipy.sparse import coo_array
except ImportError:
    from scipy.sparse import coo_matrix as coo_array

try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


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


def _load_frame_data(args):
    """Helper function to load sparse matrix data for a single frame."""
    frame, loaded = args
    row = loaded[f"row{frame}"]
    col = loaded[f"col{frame}"]
    data = loaded[f"data{frame}"]
    shape = loaded[f"shape{frame}"]
    return frame, coo_array((data, (row, col)), shape=shape)


def load_sparse(file) -> dict[int, coo_array]:
    """Load a sparse array from disk using multiprocessing."""
    with np.load(file) as loaded:
        keys = loaded.keys()
        frames = [int(key[3:]) for key in keys if key.startswith("row")]
        if not frames:
            raise ValueError("No sparse arrays found in file.")

        with Pool() as pool:
            sparse_arrs = dict(
                tqdm(
                    pool.map(_load_frame_data, [(frame, loaded) for frame in frames]),
                    total=len(frames),
                    desc="Loading adjacency arrays",
                )
            )

    return sparse_arrs


class AggregateProperties(Enum):
    """Enumeration of the properties that can be calculated for an aggregate."""

    AGGREGATION_NUMBERS = "Aggregation numbers"
    EAB = r"$e_{ab}$"
    EAC = r"$e_{ac}$"
    RADIUS_OF_GYRATION = r"Radius of gyration ($\mathrm{\AA}$)"
    VOLUME = r"Volume ($\mathrm{\AA}^3$)"
    VOL_PER_SURFACTANT = r"Volume per surfactant ($\mathrm{\AA}^3$)"
    SURFACE_AREA = r"Surface area ($\mathrm{\AA}^2$)"
    SURFACE_AREA_PER_SURFACTANT = r"Surface area per surfactant ($\mathrm{\AA}^2$)"
    SURFACE_AREA_TO_VOLUME = r"Surface area / Volume ($\mathrm{\AA}^{-1}$)"
    NORMALISED_AGGREGATION_NUMBERS = "Normalised aggregation numbers"
    TOTAL_VOLUME = r"Total excluded volume estimate ($\mathrm{\AA}^3$)"
    SOAP_VECTOR = "SOAP vector"
    SOAP_SIM_1 = "Principal component 1 of SOAP KPCA"
    SOAP_SIM_2 = "Principal component 2 of SOAP KPCA"
    VESICALITY = "Vesicality"
    COUNTERIONS_INSIDE = "Counterions inside"
    WATER_INSIDE = "Water molecules inside"
    INNER_AOT = "Inner layer AOT molecules"

    @classmethod
    def all(cls) -> 'set["AggregateProperties"]':
        return set(cls)

    @classmethod
    def fast(cls):
        return cls.all().difference(
            {
                cls.VOLUME,
                cls.VOL_PER_SURFACTANT,
                cls.SURFACE_AREA,
                cls.SURFACE_AREA_PER_SURFACTANT,
                cls.SURFACE_AREA_TO_VOLUME,
                cls.TOTAL_VOLUME,
                cls.SOAP_VECTOR,
                cls.SOAP_SIM_1,
                cls.SOAP_SIM_2,
                cls.COUNTERIONS_INSIDE,
                cls.WATER_INSIDE,
                cls.INNER_AOT,
                cls.VESICALITY,
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


class ClusteringResults(Results):
    """Store results from a clustering analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frame_counter: list[int] = []
        self.time_counter: list[int] = []
        self.agg_nums: list[int] = []
        self.norm_agg_nums: list[float] = []

        self.volume: list[float] = []
        self.surface: list[float] = []

        self.total_volume: list[float] = []

        # Coordinate pair eccentricities
        self.eabs: list[float] = []
        self.eacs: list[float] = []

        self.radii_of_gyration: list[float] = []

        self.vesicalities: list[float] = []

        # Add storage for new properties
        self.counterions_inside: list[int] = []
        self.water_inside: list[int] = []
        self.inner_aot: list[int] = []

        self.soap_vectors: list[np.ndarray] = []

    @property
    def computed(self) -> list[str]:
        """Get the computed properties."""
        return [
            "frame_counter",
            "time_counter",
            "agg_nums",
            "norm_agg_nums",
            "volume",
            "surface",
            "total_volume",
            "eabs",
            "eacs",
            "radii_of_gyration",
            "vesicalities",
            "counterions_inside",
            "water_inside",
            "inner_aot",
            "soap_vectors",
        ]


def hypersphere_center_of_mass(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate center of mass using improved hypersphere mapping to handle PBC correctly.

    This method uses a two-step process:
    1. Get initial estimate using hypersphere mapping
    2. Translate particles using this estimate and calculate true weighted center of mass

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) array of particle positions
    box_dimensions : np.ndarray
        Shape (3,) array of box dimensions
    masses : np.ndarray, optional
        Shape (N,) array of particle masses. If None, assumes equal masses.

    Returns
    -------
    np.ndarray
        Shape (3,) center of mass position
    """
    if masses is None:
        masses = np.ones(len(positions))

    # Step 1: Get initial estimate using hypersphere mapping
    # Map to hypersphere (convert to angular coordinates)
    angles = 2 * np.pi * positions / box_dimensions

    # Convert to unit vectors on hypersphere
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Average the unit vectors (weighted by mass)
    total_mass = np.sum(masses)
    mean_cos = np.sum(masses[:, np.newaxis] * cos_angles, axis=0) / total_mass
    mean_sin = np.sum(masses[:, np.newaxis] * sin_angles, axis=0) / total_mass

    # Convert back to angles
    mean_angles = np.arctan2(mean_sin, mean_cos)

    # Convert back to Cartesian coordinates for initial estimate
    pseudo_com = (mean_angles * box_dimensions) / (2 * np.pi)
    pseudo_com = np.where(pseudo_com < 0, pseudo_com + box_dimensions, pseudo_com)

    # Step 2: Translate particles to center them around the pseudo-COM
    # and calculate true center of mass
    translated_positions = positions.copy()

    for dim in range(3):
        # Calculate displacement to pseudo-COM
        displacement = positions[:, dim] - pseudo_com[dim]

        # Apply minimum image convention to handle PBC
        displacement = np.where(
            displacement > box_dimensions[dim] / 2,
            displacement - box_dimensions[dim],
            displacement,
        )
        displacement = np.where(
            displacement < -box_dimensions[dim] / 2,
            displacement + box_dimensions[dim],
            displacement,
        )

        # Translate to center around pseudo-COM (which becomes origin)
        translated_positions[:, dim] = displacement

    # Calculate true weighted center of mass from translated positions
    true_com_translated = (
        np.sum(masses[:, np.newaxis] * translated_positions, axis=0) / total_mass
    )

    # Translate back to original coordinate system
    true_com = pseudo_com + true_com_translated

    # Ensure result is within the box
    true_com = np.mod(true_com, box_dimensions)

    return true_com


def get_bead_radius(atom_type: str, coarse: bool = False) -> float:
    """Determine the bead radius for an atom type.

    Parameters
    ----------
    atom_type : str
        The atom type (first character of atom.type in MDAnalysis)
    coarse : bool, default=False
        Whether this is a coarse-grained simulation

    Returns
    -------
    float
        Bead radius in Angstroms
    """
    if coarse:
        # Coarse-grained Martini radii (see cluster.py vdwradii property)
        if atom_type == "S":
            return 0.230
        elif atom_type == "T":
            return 0.191
        else:
            return 0.264
    else:
        # For atomistic simulations, use a default radius
        # This could be refined based on actual van der Waals radii
        return 2.0  # Default radius in Angstroms


def calculate_hydrodynamic_radius(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
    atom_types: Optional[np.ndarray] = None,
    use_rpy: bool = True,
    coarse: bool = True,
) -> float:
    """Calculate hydrodynamic radius for a set of particles with PBC handling.

    Uses either Oseen kernel (1/r) or Rotne-Prager-Yamakawa kernel with:
    <R_H^-1> = (2/(N(N-1))) * sum_{i<j} <kernel(r_ij)>
    R_H = 1 / <R_H^-1>

    Only sums over the upper triangle of the distance matrix (i<j) to avoid
    double counting and exclude self-interactions.

    Uses JAX for efficient pairwise distance calculations when available,
    falls back to NumPy implementation otherwise.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) array of particle positions
    box_dimensions : np.ndarray
        Shape (3,) array of box dimensions
    atom_types : np.ndarray, optional
        Shape (N,) array of atom types for determining bead radii
    use_rpy : bool, default=True
        Whether to use Rotne-Prager-Yamakawa kernel instead of Oseen kernel
    coarse : bool, default=True
        Whether this is a coarse-grained simulation (affects bead radius calculation)

    Returns
    -------
    float
        Hydrodynamic radius in same units as positions
    """
    N = len(positions)
    if N < 2:
        return 0.0

    if JAX_AVAILABLE and N > 10:  # Use JAX for larger systems where it's more efficient
        return _calculate_hydrodynamic_radius_jax(
            positions, box_dimensions, atom_types, use_rpy, coarse
        )
    else:
        return _calculate_hydrodynamic_radius_numpy(
            positions, box_dimensions, atom_types, use_rpy, coarse
        )


if JAX_AVAILABLE:

    @partial(jit, static_argnames=["use_rpy"])
    def _compute_rh(positions, box_dimensions, bead_radii, use_rpy):
        N = len(positions)

        # Convert to JAX arrays
        pos_jax = jnp.array(positions)
        box_jax = jnp.array(box_dimensions)
        radii_jax = jnp.array(bead_radii)

        # Manually calculate all pairwise distances
        # Create indices for all pairs (i, j) where i < j (upper triangle only)
        i_indices, j_indices = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing="ij")

        # Create mask to only include upper triangle (i < j)
        mask = i_indices < j_indices

        # Vectorized pairwise distance calculation
        # Expand dimensions for broadcasting
        pos_i = pos_jax[i_indices]  # Shape: (N, N, 3)
        pos_j = pos_jax[j_indices]  # Shape: (N, N, 3)

        # Calculate displacement vectors
        displacements = pos_i - pos_j  # Shape: (N, N, 3)

        # Apply PBC to all displacements at once
        displacements -= jnp.round(displacements / box_jax) * box_jax

        # Calculate distances
        distances = jnp.sqrt(jnp.sum(displacements**2, axis=2))  # Shape: (N, N)

        # Apply mask to only include upper triangle and exclude very small distances
        valid_mask = mask & (distances > 1e-6)

        if use_rpy:
            # Rotne-Prager-Yamakawa kernel (scalar, orientationally averaged)
            # Get bead radii for each pair
            ai = radii_jax[i_indices]  # Shape: (N, N)
            aj = radii_jax[j_indices]  # Shape: (N, N)

            # Non-overlap condition: r >= ai + aj
            rcut = ai + aj
            non_overlap = distances >= rcut

            # Non-overlap formula (unequal radii): phi = (1/r) * (1 + (ai^2 + aj^2)/(3*r^2))
            kernel_non_overlap = (1.0 / distances) * (
                1.0 + (ai**2 + aj**2) / (3.0 * distances**2)
            )

            # Overlap formula (r < ai + aj): more complex, use simplified approximation
            # For simplicity, use the average radius approximation for overlap region
            a_avg = (ai + aj) / 2.0
            kernel_overlap = (1 / (2 * a_avg)) * (
                1 - (9 * distances) / (32 * a_avg) + (distances**3) / (32 * a_avg**3)
            )

            # Combine overlap and non-overlap regions
            kernel = jnp.where(non_overlap, kernel_non_overlap, kernel_overlap)
        else:
            # Oseen kernel (original)
            kernel = 1.0 / distances

        # Apply mask and calculate kernel values where valid
        kernel_values = jnp.where(valid_mask, kernel, 0.0)

        # Sum all kernel values
        kernel_sum = jnp.sum(kernel_values)

        # Calculate average kernel value
        # Now we sum over N(N-1)/2 terms instead of N²
        avg_kernel = kernel_sum / (N * (N - 1) / 2)

        # Return hydrodynamic radius
        return 1.0 / avg_kernel


def _calculate_hydrodynamic_radius_jax(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
    atom_types: Optional[np.ndarray] = None,
    use_rpy: bool = True,
    coarse: bool = True,
) -> float:
    """JAX-optimized hydrodynamic radius calculation using manual pairwise distances."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available")

    # Determine bead radii for each atom
    if atom_types is not None:
        bead_radii = np.array(
            [get_bead_radius(atom_type, coarse) for atom_type in atom_types]
        )
    else:
        # Use default radius for all atoms
        default_radius = get_bead_radius("", coarse)
        bead_radii = np.full(len(positions), default_radius)

    rh = _compute_rh(positions, box_dimensions, bead_radii, use_rpy)
    return rh if not jnp.isnan(rh) else 0.0


def _calculate_hydrodynamic_radius_numpy(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
    atom_types: Optional[np.ndarray] = None,
    use_rpy: bool = True,
    coarse: bool = True,
) -> float:
    """SciPy pdist-based implementation for hydrodynamic radius calculation."""

    N = len(positions)

    # Determine bead radii for each atom
    if atom_types is not None:
        bead_radii = np.array(
            [get_bead_radius(atom_type, coarse) for atom_type in atom_types]
        )
    else:
        # Use default radius for all atoms
        default_radius = get_bead_radius("", coarse)
        bead_radii = np.full(N, default_radius)

    # For PBC, we need to handle the minimum image convention
    # Since pdist doesn't handle PBC directly, we'll apply PBC corrections
    # to the positions first, then use pdist

    def pbc_distance_metric(u, v):
        """Custom distance metric that handles periodic boundary conditions."""
        displacement = u - v
        # Apply minimum image convention
        displacement = (
            displacement - np.round(displacement / box_dimensions) * box_dimensions
        )
        return np.linalg.norm(displacement)

    # Use pdist with custom PBC distance metric
    # pdist returns condensed distance matrix (upper triangle only)
    distances_condensed = pdist(positions, metric=pbc_distance_metric)

    # Filter out very small distances
    valid_distances = distances_condensed[distances_condensed > 1e-6]

    if use_rpy and len(valid_distances) > 0:
        # For RPY kernel, we need to map back to get the corresponding bead radii
        # Since pdist returns condensed form, we need to reconstruct which atoms correspond to each distance
        from scipy.spatial.distance import squareform

        # Get full distance matrix
        distance_matrix = squareform(distances_condensed)

        # Calculate kernel values
        kernel_values = []

        for i in range(N):
            for j in range(i + 1, N):
                rij = distance_matrix[i, j]
                if rij > 1e-6:
                    # Get individual radii for this pair
                    ai = bead_radii[i]
                    aj = bead_radii[j]

                    # Non-overlap condition: r >= ai + aj
                    rcut = ai + aj

                    if rij >= rcut:
                        # Non-overlap formula (unequal radii): phi = (1/r) * (1 + (ai^2 + aj^2)/(3*r^2))
                        kernel = (1.0 / rij) * (1.0 + (ai**2 + aj**2) / (3.0 * rij**2))
                    else:
                        # Overlap formula: use simplified approximation with average radius
                        # (More complex exact formula exists but this is reasonable approximation)
                        a_avg = (ai + aj) / 2.0
                        kernel = (1 / (2 * a_avg)) * (
                            1 - (9 * rij) / (32 * a_avg) + (rij**3) / (32 * a_avg**3)
                        )

                    kernel_values.append(kernel)

        kernel_values = np.array(kernel_values)
    else:
        # Oseen kernel (original)
        kernel_values = 1.0 / valid_distances

    # Sum all kernel values
    kernel_sum = np.sum(kernel_values)

    # Calculate average kernel value
    # We sum over N(N-1)/2 terms (upper triangle only)
    avg_kernel = kernel_sum / (N * (N - 1) / 2)

    # Return hydrodynamic radius
    return float(1.0 / avg_kernel) if avg_kernel > 0 else 0.0


def calculate_radius_of_gyration(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
    masses: Optional[np.ndarray] = None,
    center_of_mass: Optional[np.ndarray] = None,
) -> float:
    """Calculate radius of gyration for a set of particles with PBC handling.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) array of particle positions
    box_dimensions : np.ndarray
        Shape (3,) array of box dimensions
    masses : np.ndarray, optional
        Shape (N,) array of particle masses. If None, assumes equal masses.
    center_of_mass : np.ndarray, optional
        Shape (3,) center of mass position. If None, will be calculated.

    Returns
    -------
    float
        Radius of gyration in same units as positions
    """
    if masses is None:
        masses = np.ones(len(positions))

    # Calculate center of mass if not provided
    if center_of_mass is None:
        center_of_mass = hypersphere_center_of_mass(positions, box_dimensions, masses)

    # Calculate distances from center of mass with PBC
    distances_squared = np.zeros(len(positions))
    total_mass = np.sum(masses)

    for i, pos in enumerate(positions):
        # Calculate displacement vector from COM
        displacement = pos - center_of_mass

        # Apply minimum image convention for PBC
        for dim in range(3):
            if displacement[dim] > box_dimensions[dim] / 2:
                displacement[dim] -= box_dimensions[dim]
            elif displacement[dim] < -box_dimensions[dim] / 2:
                displacement[dim] += box_dimensions[dim]

        distances_squared[i] = np.sum(displacement**2)

    # Calculate weighted average of squared distances
    rg_squared = np.sum(masses * distances_squared) / total_mass

    return np.sqrt(rg_squared)
