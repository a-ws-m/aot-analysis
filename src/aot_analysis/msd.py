"""Utilities for tracking aggregate lifetimes across molecular dynamics simulations.

This module provides functionality to analyze the lifetime of molecular aggregates
by tracking their formation, persistence, and dissolution over time. A unique
aggregate is defined by its constituent molecules, and lifetimes track concurrent
timesteps where the aggregate exists.
"""

from collections import defaultdict
from pathlib import Path
from typing import FrozenSet, Optional, Set

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis.base import AnalysisBase
from scipy import stats
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from scipy.spatial.distance import pdist, squareform

from .cluster import get_adj_array

TIMESCALE_CUTOFF = 2500  # ps


class AggregateLifetimeTracker(AnalysisBase):
    """Class for tracking aggregate lifetimes across MD trajectories.

    This analysis identifies molecular aggregates at each frame and tracks their
    lifetimes by monitoring when the same set of molecules forms aggregates
    across consecutive timesteps.

    Parameters
    ----------
    tailgroups : MDAnalysis.AtomGroup
        Atoms used for clustering (typically tail atoms of surfactants)
    cutoff : float, default=4.25
        Distance cutoff for adjacency matrix calculation (Angstroms)
    min_cluster_size : int, default=5
        Minimum number of molecules required to define an aggregate
    verbose : bool, default=True
        Whether to print progress information
    """

    def __init__(
        self,
        tailgroups: mda.AtomGroup,
        cutoff: float = 4.25,
        min_cluster_size: int = 5,
        verbose: bool = True,
        **kwargs,
    ):
        trajectory = tailgroups.universe.trajectory
        super().__init__(trajectory, verbose, **kwargs)

        self.tailgroups = tailgroups
        self.cutoff = cutoff
        self.min_cluster_size = min_cluster_size

        # Calculate molecule information
        self.num_surf = tailgroups.n_residues
        self.whole_molecules = tailgroups.residues.unique
        self.atom_per_mol = int(len(tailgroups) / self.num_surf)

        # Sort tailgroups by residue number for consistent indexing
        tailgroups_ = tailgroups.universe.atoms[[]]
        for residue in self.whole_molecules:
            tailgroups_ += residue.atoms & tailgroups
        self.tailgroups = tailgroups_

        # Storage for tracking aggregates over time
        self.active_aggregates: dict[FrozenSet[int], dict] = {}
        self.completed_aggregates: list[dict] = []

        # Results storage
        self.results = {}

    def _single_frame(self):
        """Process a single frame to identify aggregates and update lifetimes."""
        current_frame = int(self._ts.frame)
        current_time = self._ts.time

        # Calculate adjacency matrix for current frame (molecule-level)
        # get_adj_array internally uses atom_to_mol_pairs to convert atom contacts
        # to molecule contacts, so the resulting matrix is (num_molecules, num_molecules)
        sparse_adj_arr = get_adj_array(
            self.tailgroups, self.cutoff, self._ts.dimensions
        )

        # Find connected components (aggregates)
        # connected_comps contains molecule indices, not atom indices
        n_aggregates, connected_comps = connected_components(
            sparse_adj_arr, directed=False
        )

        # Identify current aggregates (as sets of molecule indices)
        # Note: connected_comps already contains molecule-level clustering because
        # get_adj_array internally converts atom pairs to molecule pairs
        current_aggregates: Set[FrozenSet[int]] = set()

        for i in range(n_aggregates):
            # Get molecule indices for this aggregate
            molecule_indices = np.where(connected_comps == i)[0]
            agg_size = len(molecule_indices)

            # Only consider aggregates above minimum size
            if agg_size >= self.min_cluster_size:
                current_aggregates.add(frozenset(molecule_indices))

        # Update aggregate lifetimes
        self._update_aggregate_lifetimes(
            current_aggregates, current_frame, current_time
        )

    def _update_aggregate_lifetimes(
        self, current_aggregates: Set[FrozenSet[int]], frame: int, time: float
    ):
        """Update the lifetime tracking for aggregates."""
        # Check which active aggregates are still present
        still_active = set()

        for agg_molecules in current_aggregates:
            if agg_molecules in self.active_aggregates:
                # Aggregate continues to exist - update end time
                self.active_aggregates[agg_molecules]["end_frame"] = frame
                self.active_aggregates[agg_molecules]["end_time"] = time
                still_active.add(agg_molecules)
            else:
                # New aggregate formed
                self.active_aggregates[agg_molecules] = {
                    "molecules": agg_molecules,
                    "aggregation_number": len(agg_molecules),
                    "start_frame": frame,
                    "start_time": time,
                    "end_frame": frame,
                    "end_time": time,
                }
                still_active.add(agg_molecules)

        # Move aggregates that are no longer active to completed list
        to_remove = []
        for agg_molecules, agg_data in self.active_aggregates.items():
            if agg_molecules not in still_active:
                self.completed_aggregates.append(agg_data.copy())
                to_remove.append(agg_molecules)

        # Remove completed aggregates from active tracking
        for agg_molecules in to_remove:
            del self.active_aggregates[agg_molecules]

    def _conclude(self):
        """Finalize analysis and create results DataFrame."""
        # Move any remaining active aggregates to completed
        for agg_data in self.active_aggregates.values():
            self.completed_aggregates.append(agg_data.copy())

        # Create DataFrame from completed aggregates
        if self.completed_aggregates:
            df_data = []
            for i, agg_data in enumerate(self.completed_aggregates):
                # Convert frozenset to sorted tuple for consistent indexing
                molecules_tuple = tuple(sorted(agg_data["molecules"]))

                df_data.append(
                    {
                        "aggregate_id": i,
                        "molecules": molecules_tuple,
                        "aggregation_number": agg_data["aggregation_number"],
                        "start_frame": agg_data["start_frame"],
                        "start_time": agg_data["start_time"],
                        "end_frame": agg_data["end_frame"],
                        "end_time": agg_data["end_time"],
                        "lifetime_frames": agg_data["end_frame"]
                        - agg_data["start_frame"]
                        + 1,
                        "lifetime_time": agg_data["end_time"] - agg_data["start_time"],
                    }
                )

            self.results_df = pd.DataFrame(df_data)

            # Set multi-index as requested (molecules, start_frame)
            self.results_df = self.results_df.set_index(["molecules", "start_frame"])

        else:
            # Create empty DataFrame with proper structure
            self.results_df = pd.DataFrame(
                columns=[
                    "aggregate_id",
                    "aggregation_number",
                    "start_time",
                    "end_frame",
                    "end_time",
                    "lifetime_frames",
                    "lifetime_time",
                ]
            )
            self.results_df.index = pd.MultiIndex.from_tuples(
                [], names=["molecules", "start_frame"]
            )

    def save_results(self, filepath: Path):
        """Save the results DataFrame to a CSV file."""
        self.results_df.to_csv(filepath)

    @property
    def lifetime_dataframe(self) -> pd.DataFrame:
        """Access the results DataFrame containing aggregate lifetimes."""
        return self.results_df


def load_lifetime_dataframe(filepath: str) -> pd.DataFrame:
    """Load aggregate lifetime DataFrame from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing lifetime data

    Returns
    -------
    pd.DataFrame
        DataFrame with aggregate lifetimes, indexed by (molecules, start_frame)
    """
    df = pd.read_csv(filepath, index_col=[0, 1])
    # Convert the molecules column from string representation back to tuple
    if "molecules" in df.columns:
        df["molecules"] = df["molecules"].apply(eval)
    return df


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


def calculate_hydrodynamic_radius(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
) -> float:
    """Calculate hydrodynamic radius for a set of particles with PBC handling.

    Uses the formula: <R_H^-1> = (2/(N(N-1))) * sum_{i<j} <1/|r_i - r_j|>
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

    Returns
    -------
    float
        Hydrodynamic radius in same units as positions
    """
    N = len(positions)
    if N < 2:
        return 0.0

    if JAX_AVAILABLE and N > 10:  # Use JAX for larger systems where it's more efficient
        return _calculate_hydrodynamic_radius_jax(positions, box_dimensions)
    else:
        return _calculate_hydrodynamic_radius_numpy(positions, box_dimensions)


if JAX_AVAILABLE:

    @jit
    def _compute_rh(positions, box_dimensions):
        N = len(positions)

        # Convert to JAX arrays
        pos_jax = jnp.array(positions)
        box_jax = jnp.array(box_dimensions)

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

        # Calculate inverse distances where mask is True, 0 elsewhere
        inverse_distances = jnp.where(valid_mask, 1.0 / distances, 0.0)

        # Sum all inverse distances
        inverse_r_sum = jnp.sum(inverse_distances)

        # Calculate average inverse distance
        # Now we sum over N(N-1)/2 terms instead of N²
        avg_inverse_r = inverse_r_sum / (N * (N - 1) / 2)

        # Return hydrodynamic radius
        return 1.0 / avg_inverse_r


def _calculate_hydrodynamic_radius_jax(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
) -> float:
    """JAX-optimized hydrodynamic radius calculation using manual pairwise distances."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available")

    rh = _compute_rh(positions, box_dimensions)
    return rh if not jnp.isnan(rh) else 0.0


def _calculate_hydrodynamic_radius_numpy(
    positions: np.ndarray,
    box_dimensions: np.ndarray,
) -> float:
    """SciPy pdist-based implementation for hydrodynamic radius calculation."""

    N = len(positions)

    # For PBC, we need to handle the minimum image convention
    # Since pdist doesn't handle PBC directly, we'll apply PBC corrections
    # to the positions first, then use pdist

    def pbc_distance_metric(u, v):
        """Custom distance metric that handles periodic boundary conditions."""
        displacement = u - v
        # Apply minimum image convention
        displacement -= np.round(displacement / box_dimensions) * box_dimensions
        return np.linalg.norm(displacement)

    # Use pdist with custom PBC distance metric
    # pdist returns condensed distance matrix (upper triangle only)
    distances_condensed = pdist(positions, metric=pbc_distance_metric)

    # Filter out very small distances
    valid_distances = distances_condensed[distances_condensed > 1e-6]

    # Calculate inverse distances for valid distances only
    inverse_distances = 1.0 / valid_distances

    # Sum all inverse distances
    inverse_r_sum = np.sum(inverse_distances)

    # Calculate average inverse distance
    # We sum over N(N-1)/2 terms (upper triangle only)
    avg_inverse_r = inverse_r_sum / (N * (N - 1) / 2)

    # Return hydrodynamic radius
    return float(1.0 / avg_inverse_r) if avg_inverse_r > 0 else 0.0


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


def analyze_aggregate_lifetimes(
    trajectory_path: str,
    structure_path: str,
    cutoff: float = 4.25,
    min_cluster_size: int = 5,
    tail_selection: str = "name C6 C7 C8 C9 C10 C11 C15 C16 C17 C18 C19 C20",
    step: int = 1,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyze aggregate lifetimes for a molecular dynamics trajectory.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file
    structure_path : str
        Path to the structure/topology file
    cutoff : float, default=4.25
        Distance cutoff for clustering (Angstroms)
    min_cluster_size : int, default=5
        Minimum number of molecules to define an aggregate
    tail_selection : str
        MDAnalysis selection string for tail atoms used in clustering
    step : int, default=1
        Step size for trajectory analysis
    start : int, optional
        Starting frame for analysis
    stop : int, optional
        Ending frame for analysis
    output_path : str, optional
        Path to save results CSV file
    verbose : bool, default=True
        Whether to print progress information

    Returns
    -------
    pd.DataFrame
        DataFrame with aggregate lifetimes, indexed by (molecules, start_frame)
    """
    # Load trajectory
    u = mda.Universe(structure_path, trajectory_path)

    # Select tail atoms for clustering
    tailgroups = u.select_atoms(tail_selection)

    if verbose:
        print(
            f"Selected {len(tailgroups)} tail atoms from {tailgroups.n_residues} molecules"
        )
        print(f"Using cutoff distance: {cutoff} Å")
        print(f"Minimum cluster size: {min_cluster_size} molecules")

    # Run lifetime analysis
    tracker = AggregateLifetimeTracker(
        tailgroups=tailgroups,
        cutoff=cutoff,
        min_cluster_size=min_cluster_size,
        verbose=verbose,
    )

    # Handle optional parameters for MDAnalysis run method
    if start is not None and stop is not None:
        tracker.run(start=start, stop=stop, step=step)
    elif start is not None:
        tracker.run(start=start, step=step)
    elif stop is not None:
        tracker.run(stop=stop, step=step)
    else:
        tracker.run(step=step)

    # Save results if output path provided
    if output_path:
        output_file = Path(output_path)
        tracker.save_results(output_file)
        if verbose:
            print(f"Results saved to: {output_file}")

    return tracker.lifetime_dataframe


def calculate_aggregate_sd(
    trajectory_path: str,
    structure_path: str,
    lifetime_df: pd.DataFrame,
    tail_selection: str = "name C6 C7 C8 C9 C10 C11 C15 C16 C17 C18 C19 C20",
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculate squared displacement for each aggregate over its lifetime.

    Uses a sliding window approach to calculate SD within TIMESCALE_CUTOFF
    windows throughout each aggregate's lifetime, maximizing data utilization
    while avoiding artifacts from very long time scales.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file
    structure_path : str
        Path to the structure/topology file
    lifetime_df : pd.DataFrame
        DataFrame with aggregate lifetimes from analyze_aggregate_lifetimes
    tail_selection : str
        MDAnalysis selection string for tail atoms
    verbose : bool
        Whether to print progress information

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: aggregation_number, delta_t, sd
    """
    # Load trajectory
    u = mda.Universe(structure_path, trajectory_path)
    tailgroups = u.select_atoms(tail_selection)

    # Create mapping from molecule index to residue
    residue_mapping = {}
    for i, residue in enumerate(tailgroups.residues.unique):
        residue_mapping[i] = residue

    sd_data = []

    if verbose:
        print(f"Calculating SD for {len(lifetime_df)} aggregates...")

    if len(lifetime_df) == 0:
        if verbose:
            print("Warning: No aggregate lifetime data provided")
        return pd.DataFrame()

    # Process each unique aggregate
    for idx, (molecules_tuple, start_frame) in tqdm(
        enumerate(lifetime_df.index),
        desc="Processing aggregate trajectories",
        total=len(lifetime_df.index),
    ):
        agg_data = lifetime_df.loc[(molecules_tuple, start_frame)]

        # Get molecule indices and aggregate info
        molecule_indices = (
            molecules_tuple
            if isinstance(molecules_tuple, tuple)
            else eval(molecules_tuple)
        )

        # Extract values, handling potential Series indexing issues
        try:
            agg_size = int(agg_data["aggregation_number"])  # type: ignore
            end_frame_val = int(agg_data["end_frame"])  # type: ignore
        except (KeyError, AttributeError, IndexError, TypeError) as e:
            if verbose:
                print(f"Warning: Could not extract data for aggregate {idx}: {e}")
            continue

        start_frame_val = int(start_frame)

        if not end_frame_val > start_frame_val:
            # Only exists for one frame, skip
            continue

        # Get the residues for this aggregate
        agg_residues = [residue_mapping[mol_idx] for mol_idx in molecule_indices]

        # Get positions of all atoms in the aggregate
        # Start with the first residue's atoms
        agg_atoms = agg_residues[0].atoms
        for residue in agg_residues[1:]:
            agg_atoms += residue.atoms

        # Calculate center of mass trajectory for this aggregate
        com_trajectory = []
        times = []

        for frame_idx in range(start_frame_val, end_frame_val + 1):
            u.trajectory[frame_idx]
            dt = u.trajectory.dt

            # Calculate center of mass using improved hypersphere method with masses
            com = hypersphere_center_of_mass(
                agg_atoms.positions, u.dimensions[:3], agg_atoms.masses
            )

            com_trajectory.append(com)
            times.append(times[-1] + dt if times else u.trajectory.time)

        com_trajectory = np.array(com_trajectory)
        times = np.array(times)

        # Get the trajectory timestep
        timestep = u.trajectory.dt

        # Calculate maximum frames that fit within TIMESCALE_CUTOFF
        max_dt_frames = (
            int(TIMESCALE_CUTOFF / abs(timestep))
            if timestep != 0
            else len(com_trajectory) - 1
        )

        # Use sliding windows within TIMESCALE_CUTOFF duration
        n_frames = len(com_trajectory)

        # Calculate SD using sliding windows approach
        for dt_frames in range(1, min(max_dt_frames + 1, n_frames)):
            sd_values = []

            # Use sliding windows throughout the trajectory
            for start_idx in range(n_frames - dt_frames):
                end_idx = start_idx + dt_frames

                # Calculate squared displacement
                displacement = com_trajectory[end_idx] - com_trajectory[start_idx]

                # Handle PBC for displacement
                box_dims = u.dimensions[:3]
                for dim in range(3):
                    if displacement[dim] > box_dims[dim] / 2:
                        displacement[dim] -= box_dims[dim]
                    elif displacement[dim] < -box_dims[dim] / 2:
                        displacement[dim] += box_dims[dim]

                sd_values.append(np.sum(displacement**2))

            if sd_values:  # Only add if we have valid SD values
                # Calculate actual delta_t for this dt_frames separation
                delta_t = dt_frames * abs(timestep)

                mean_sd = np.mean(sd_values)

                sd_data.append(
                    {
                        "aggregate_id": idx,
                        "aggregation_number": agg_size,
                        "delta_t": delta_t,
                        "sd": mean_sd,
                        "n_samples": len(sd_values),
                    }
                )

    return pd.DataFrame(sd_data)


def calculate_aggregate_hydrodynamic_radius(
    trajectory_path: str,
    structure_path: str,
    lifetime_df: pd.DataFrame,
    tail_selection: str = "name C6 C7 C8 C9 C10 C11 C15 C16 C17 C18 C19 C20",
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculate average hydrodynamic radius for aggregates by size.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file
    structure_path : str
        Path to the structure/topology file
    lifetime_df : pd.DataFrame
        DataFrame with aggregate lifetimes from analyze_aggregate_lifetimes
    tail_selection : str
        MDAnalysis selection string for tail atoms
    verbose : bool
        Whether to print progress information

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: aggregation_number, hydrodynamic_radius_avg, hydrodynamic_radius_std, n_aggregates
    """
    # Load trajectory
    u = mda.Universe(structure_path, trajectory_path)
    tailgroups = u.select_atoms(tail_selection)

    # Create mapping from molecule index to residue
    residue_mapping = {}
    for i, residue in enumerate(tailgroups.residues.unique):
        residue_mapping[i] = residue

    # Store hydrodynamic radius data by aggregation number
    rh_by_size = defaultdict(list)

    if verbose:
        print(f"Calculating hydrodynamic radius for {len(lifetime_df)} aggregates...")

    if len(lifetime_df) == 0:
        if verbose:
            print("Warning: No aggregate lifetime data provided")
        return pd.DataFrame()

    # Process each unique aggregate
    for idx, (molecules_tuple, start_frame) in tqdm(
        enumerate(lifetime_df.index),
        desc="Processing aggregate hydrodynamic radius",
        total=len(lifetime_df.index),
    ):
        agg_data = lifetime_df.loc[(molecules_tuple, start_frame)]

        # Get molecule indices and aggregate info
        molecule_indices = (
            molecules_tuple
            if isinstance(molecules_tuple, tuple)
            else eval(molecules_tuple)
        )

        # Extract values, handling potential Series indexing issues
        try:
            agg_size = int(agg_data["aggregation_number"])  # type: ignore
            end_frame_val = int(agg_data["end_frame"])  # type: ignore
        except (KeyError, AttributeError, IndexError, TypeError) as e:
            if verbose:
                print(f"Warning: Could not extract data for aggregate {idx}: {e}")
            continue

        start_frame_val = int(start_frame)

        # Get the residues for this aggregate
        agg_residues = [residue_mapping[mol_idx] for mol_idx in molecule_indices]

        # Get positions of all atoms in the aggregate
        # Start with the first residue's atoms
        agg_atoms = agg_residues[0].atoms
        for residue in agg_residues[1:]:
            agg_atoms += residue.atoms

        # We'll use all atom positions for hydrodynamic radius calculation
        rh_values = []

        # Sample frames throughout the aggregate's lifetime
        total_frames = end_frame_val - start_frame_val + 1
        # Sample every 500 frames or at least 5 frames, but don't exceed total frames
        sample_interval = max(1, total_frames // 5)
        sample_frames = range(start_frame_val, end_frame_val + 1, sample_interval)

        for frame_idx in sample_frames:
            u.trajectory[frame_idx]

            # Calculate hydrodynamic radius using all atom positions in the aggregate
            rh = calculate_hydrodynamic_radius(agg_atoms.positions, u.dimensions[:3])

            if rh > 0:  # Only add valid values
                rh_values.append(rh)

        if rh_values:  # Only add if we have valid Rh values
            # Calculate average hydrodynamic radius for this aggregate
            avg_rh = np.mean(rh_values)
            rh_by_size[agg_size].append(avg_rh)

    # Create summary DataFrame
    rh_summary_data = []
    for agg_size, rh_values in rh_by_size.items():
        if rh_values:
            rh_summary_data.append(
                {
                    "aggregation_number": agg_size,
                    "hydrodynamic_radius_avg": np.mean(rh_values),
                    "hydrodynamic_radius_std": np.std(rh_values),
                    "n_aggregates": len(rh_values),
                }
            )

    return pd.DataFrame(rh_summary_data)


def estimate_diffusion_coefficients(
    sd_df: pd.DataFrame, verbose: bool = False, filter_: bool = True
) -> pd.DataFrame:
    """Estimate diffusion coefficients from SD data using Einstein relation.

    D = SD / (6 * delta_t) for 3D diffusion

    Uses weighted least squares regression with weights = 1/var(SD) to account
    for increasing variance at longer time scales. SD data is already constrained
    to reasonable time scales by the sliding window approach in calculate_aggregate_sd.

    Parameters
    ----------
    sd_df : pd.DataFrame
        DataFrame with SD data from calculate_aggregate_sd
    verbose : bool
        Whether to print debugging information
    filter_: bool
        Whether to apply additional filtering for time scales (mainly for fine-tuning)

    Returns
    -------
    pd.DataFrame
        DataFrame with diffusion coefficients by aggregate size
    """
    diffusion_data = []

    if len(sd_df) == 0:
        if verbose:
            print("Warning: Empty SD DataFrame provided")
        return pd.DataFrame()

    # Group by aggregation number
    n_groups = len(sd_df.groupby("aggregation_number"))
    if verbose:
        print(f"Processing {n_groups} different aggregate sizes...")

    for agg_size, group in sd_df.groupby("aggregation_number"):
        # Additional filtering for time scales where diffusion behavior is expected to be linear
        # MSD calculation already limits delta_t to TIMESCALE_CUTOFF, but this allows fine-tuning
        if filter_:
            filtered_group = group[
                (group["delta_t"] >= 1) & (group["delta_t"] <= TIMESCALE_CUTOFF)
            ]
        else:
            filtered_group = group

        if verbose:
            print(
                f"Size {agg_size}: {len(group)} total points, {len(filtered_group)} after filtering"
            )

        if len(filtered_group) < 5:  # Need sufficient data points
            if verbose:
                print(
                    f"  Skipping size {agg_size}: insufficient data points ({len(filtered_group)} < 5)"
                )
            continue

        # Calculate time-dependent variance of SD across all aggregates of this size
        # Group by delta_t and calculate variance at each time point
        time_variance = filtered_group.groupby("delta_t")["sd"].var()

        # For time points with only one measurement, use a default variance
        time_variance = time_variance.fillna(time_variance.median())

        # Create a mapping from delta_t to variance
        variance_map = time_variance.to_dict()

        # Use weighted least squares regression with weights = 1/var(SD) at each time
        try:
            X = np.array(filtered_group["delta_t"].values).reshape(
                -1, 1
            )  # reshape for sklearn
            y = np.array(filtered_group["sd"].values)

            # Calculate weights as inverse of time-dependent variance
            # Map each delta_t to its corresponding variance
            variances = np.array([variance_map[dt] for dt in filtered_group["delta_t"]])
            # Add small epsilon to avoid division by zero
            weights = 1.0 / (variances + 1e-12)

            # Fit weighted linear regression: SD = 6D * delta_t
            model = LinearRegression()
            model.fit(X, y, sample_weight=weights)

            # Calculate R-squared for weighted regression
            y_pred = model.predict(X)
            r_squared = r2_score(y, y_pred, sample_weight=weights)

            # Calculate standard error of the slope
            # For weighted least squares, this is more complex
            residuals = y - y_pred
            mse = mean_squared_error(y, y_pred, sample_weight=weights)
            X_centered = X - np.average(X, weights=weights, axis=0)
            var_slope = mse / np.sum(weights * X_centered**2)
            std_err = np.sqrt(var_slope)

            slope = model.coef_[0]
            diffusion_coeff = slope / 6.0  # Convert to diffusion coefficient
            diffusion_err = std_err / 6.0  # Error in diffusion coefficient

            diffusion_data.append(
                {
                    "aggregation_number": agg_size,
                    "diffusion_coefficient": diffusion_coeff,
                    "diffusion_error": diffusion_err,
                    "r_squared": r_squared,
                    "slope": slope,
                    "intercept": model.intercept_,
                    "n_points": len(filtered_group),
                }
            )
        except (ValueError, TypeError) as e:
            # Skip this aggregate size if regression fails
            print(f"Warning: Could not fit regression for size {agg_size}: {e}")
            continue

    return pd.DataFrame(diffusion_data)


def plot_sd_fit(
    sd_df: pd.DataFrame,
    diffusion_df: pd.DataFrame,
    aggregate_size: int = 180,
    output_dir: str = ".",
    show_plots: bool = True,
    filter_: bool = True,
):
    """Plot SD distribution vs time with fitted line for a specific aggregate size.

    Uses violin plots to show the distribution of SD values at each time point,
    with the weighted linear fit overlaid on top.

    Parameters
    ----------
    sd_df : pd.DataFrame
        SD data from calculate_aggregate_sd
    diffusion_df : pd.DataFrame
        Diffusion coefficient data from estimate_diffusion_coefficients
    aggregate_size : int
        Specific aggregate size to plot (default: 180)
    output_dir : str
        Directory to save plots
    show_plots : bool
        Whether to display plots
    filter_ : bool
        Whether to filter out long and short time scales.
    """
    # Filter data for the specific aggregate size
    size_data = sd_df[sd_df["aggregation_number"] == aggregate_size]

    if len(size_data) == 0:
        print(f"Warning: No data found for aggregate size {aggregate_size}")
        return

    # Apply time filtering if requested
    if filter_:
        filtered_data = size_data[
            (size_data["delta_t"] >= 1) & (size_data["delta_t"] <= TIMESCALE_CUTOFF)
        ]
    else:
        filtered_data = size_data

    if len(filtered_data) == 0:
        print(f"Warning: No data for aggregate size {aggregate_size} after filtering")
        return

    # Get the diffusion coefficient for this size
    size_diffusion = diffusion_df[diffusion_df["aggregation_number"] == aggregate_size]

    if len(size_diffusion) == 0:
        print(
            f"Warning: No diffusion coefficient found for aggregate size {aggregate_size}"
        )
        return

    # Extract fit parameters
    slope = size_diffusion.iloc[0]["slope"]
    intercept = size_diffusion.iloc[0]["intercept"]
    r_squared = size_diffusion.iloc[0]["r_squared"]
    diffusion_coeff = size_diffusion.iloc[0]["diffusion_coefficient"]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Create violin plot to show distribution of SD values at each time point
    sns.violinplot(
        data=filtered_data,
        x="delta_t",
        y="sd",
        native_scale=True,
        alpha=0.7,
        common_norm=False,
        color="lightblue",
        inner="box",
    )

    # Plot fitted line on top
    t_fit = np.linspace(
        filtered_data["delta_t"].min(), filtered_data["delta_t"].max(), 100
    )
    sd_fit = slope * t_fit + intercept

    # Convert diffusion coefficient to m²/s for display
    diffusion_coeff_si = diffusion_coeff * 1e-8  # Å²/ps to m²/s

    plt.plot(
        t_fit,
        sd_fit,
        "r-",
        linewidth=3,
        label=f"Weighted fit: D = {diffusion_coeff_si:.2e} m²/s",
        zorder=10,
    )

    plt.xlabel("Δt (ps)")
    plt.ylabel("SD (Å²)")
    plt.title(f"SD Distribution vs Time for Aggregate Size {aggregate_size}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add R² annotation
    plt.text(
        0.05,
        0.95,
        f"R² = {r_squared:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    if output_dir:
        plt.savefig(
            f"{output_dir}/sd_fit_size_{aggregate_size}.png",
            dpi=300,
            bbox_inches="tight",
        )

    if show_plots:
        plt.show()


def plot_sd_analysis(
    sd_df: pd.DataFrame,
    diffusion_df: pd.DataFrame,
    output_dir: str = ".",
    show_plots: bool = True,
    include_r_squared: bool = False,
    filter_: bool = True,
):
    """Create plots for SD analysis and diffusion coefficients.

    Parameters
    ----------
    sd_df : pd.DataFrame
        SD data from calculate_aggregate_sd
    diffusion_df : pd.DataFrame
        Diffusion coefficient data from estimate_diffusion_coefficients
    output_dir : str
        Directory to save plots
    show_plots : bool
        Whether to display plots
    include_r_squared : bool
        Whether to annotate diffusion plot with R² values.
    filter_ : bool
        Whether to filter out long and short time scales in SD plot.

    """
    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Plot 1: SD vs delta_t colored by aggregate size
    if len(sd_df) > 0:
        plt.figure(figsize=(10, 6))

        # Filter data for better visualization (limit aggregate sizes for readability)
        # plot_data = sd_df[sd_df["aggregation_number"] <= 20]
        if filter_:
            plot_data = sd_df[
                (sd_df["delta_t"] >= 1) & (sd_df["delta_t"] <= TIMESCALE_CUTOFF)
            ]
        else:
            plot_data = sd_df

        if len(plot_data) > 0:
            g = sns.relplot(
                data=plot_data,
                x="delta_t",
                y="sd",
                hue="aggregation_number",
                kind="line",
                height=6,
                aspect=1.5,
                alpha=0.7,
            )

            g.set_axis_labels("Δt (ps)", "SD (Å$^2$)")
            g.set(yscale="log")
            g.figure.suptitle("Squared Displacement vs Time by Aggregate Size")
            plt.tight_layout()

            if output_dir:
                plt.savefig(
                    f"{output_dir}/sd_vs_time.png", dpi=300, bbox_inches="tight"
                )
            if show_plots:
                plt.show()
        else:
            print("Warning: No SD data in the specified range for plotting")
    else:
        print("Warning: No SD data available for plotting")

    # Plot 2: Diffusion coefficient vs aggregate size
    if len(diffusion_df) > 0 and "aggregation_number" in diffusion_df.columns:
        plt.figure(figsize=(8, 6))

        # Convert diffusion coefficients to m²/s for display
        diffusion_coeff_si = (
            diffusion_df["diffusion_coefficient"] * 1e-8
        )  # Å²/ps to m²/s
        diffusion_error_si = diffusion_df["diffusion_error"] * 1e-8  # Å²/ps to m²/s

        plt.errorbar(
            diffusion_df["aggregation_number"],
            diffusion_coeff_si,
            yerr=diffusion_error_si,
            fmt="o-",
            capsize=5,
            capthick=2,
            alpha=0.8,
        )

        plt.xlabel("Aggregation Number")
        plt.ylabel("Diffusion Coefficient (m²/s)")
        plt.title("Diffusion Coefficient vs Aggregate Size")
        plt.grid(True, alpha=0.3)

        if include_r_squared:
            # Add R² values as text annotations
            for _, row in diffusion_df.iterrows():
                # Convert diffusion coefficient to SI units for annotation position
                diffusion_coeff_si_val = row["diffusion_coefficient"] * 1e-8
                plt.annotate(
                    f'R²={row["r_squared"]:.3f}',
                    (row["aggregation_number"], diffusion_coeff_si_val),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        plt.gca().set_yscale("log")
        plt.tight_layout()

        if output_dir:
            plt.savefig(
                f"{output_dir}/diffusion_vs_size.png", dpi=300, bbox_inches="tight"
            )
        if show_plots:
            plt.show()
    else:
        print("Warning: No diffusion coefficient data available for plotting")
        print("This could be due to:")
        print("- Insufficient data points for regression analysis")
        print("- All regression fits failed")
        print("- Aggregates too short-lived for meaningful MSD calculation")


def plot_hydrodynamic_radius_analysis(
    rh_df: pd.DataFrame,
    diffusion_df: Optional[pd.DataFrame] = None,
    output_dir: str = ".",
    show_plots: bool = True,
    temperature: float = 298.15,  # K
    fix_exponent: bool = False,
):
    """Create plots for hydrodynamic radius analysis and calculate effective viscosity.

    Parameters
    ----------
    rh_df : pd.DataFrame
        Hydrodynamic radius data from calculate_aggregate_hydrodynamic_radius
    diffusion_df : pd.DataFrame, optional
        Diffusion coefficient data from estimate_diffusion_coefficients
    output_dir : str
        Directory to save plots
    show_plots : bool
        Whether to display plots
    temperature : float, default=298.15
        Temperature in Kelvin for viscosity calculation
    fix_exponent : bool, default=False
        If True, fix the Stokes-Einstein exponent to -1 and use HuberRegressor for outlier handling
    """
    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Constants for viscosity calculation
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    # Plot 1: Hydrodynamic radius vs aggregation number
    if len(rh_df) > 0:
        plt.figure(figsize=(8, 6))

        plt.errorbar(
            rh_df["aggregation_number"],
            rh_df["hydrodynamic_radius_avg"],
            yerr=rh_df["hydrodynamic_radius_std"],
            fmt="o-",
            capsize=5,
            capthick=2,
            alpha=0.8,
            label="Hydrodynamic radius",
        )

        plt.xlabel("Aggregation Number")
        plt.ylabel("Hydrodynamic Radius (Å)")
        plt.title("Hydrodynamic Radius vs Aggregate Size")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if output_dir:
            plt.savefig(f"{output_dir}/rh_vs_size.png", dpi=300, bbox_inches="tight")
        if show_plots:
            plt.show()

    # Plot 2: Diffusion coefficient vs hydrodynamic radius with Stokes-Einstein fit
    if diffusion_df is not None and len(diffusion_df) > 0 and len(rh_df) > 0:
        # Merge the dataframes on aggregation number
        merged_df = pd.merge(diffusion_df, rh_df, on="aggregation_number", how="inner")

        if len(merged_df) > 0:
            plt.figure(figsize=(10, 6))

            # Convert diffusion coefficients to m²/s for display
            diffusion_coeff_si = (
                merged_df["diffusion_coefficient"] * 1e-8
            )  # Å²/ps to m²/s
            diffusion_error_si = merged_df["diffusion_error"] * 1e-8  # Å²/ps to m²/s

            # Convert hydrodynamic radius to meters for viscosity calculation
            rh_m = merged_df["hydrodynamic_radius_avg"] * 1e-10  # Å to m

            # Plot data points with error bars
            plt.errorbar(
                merged_df["hydrodynamic_radius_avg"],
                diffusion_coeff_si,
                xerr=merged_df["hydrodynamic_radius_std"],
                yerr=diffusion_error_si,
                fmt="o",
                capsize=5,
                capthick=2,
                alpha=0.8,
                label="Data",
            )

            # Fit modified Stokes-Einstein relationship: D = A * R_H^n
            # Using log-linear fitting: log(D) = log(A) + n * log(R_H)
            # This allows the exponent n to vary from the classical -1
            if len(merged_df) >= 3:  # Need at least 3 points for fitting
                try:
                    rh_values = np.array(merged_df["hydrodynamic_radius_avg"])  # Å
                    d_values = np.array(merged_df["diffusion_coefficient"])  # Å²/ps

                    # Convert to SI units for viscosity calculation
                    rh_values_m = rh_values * 1e-10  # Å to m
                    d_values_si = d_values * 1e-8  # Å²/ps to m²/s

                    # Filter out any non-positive values
                    valid_mask = (rh_values > 0) & (d_values > 0)
                    if np.sum(valid_mask) >= 3:
                        rh_fit = rh_values[valid_mask]
                        d_fit = d_values[valid_mask]
                        rh_fit_m = rh_values_m[valid_mask]
                        d_fit_si = d_values_si[valid_mask]

                        if fix_exponent:
                            # Fixed exponent case: D = A / R_H (exponent = -1)
                            # Transform to: D * R_H = A
                            # Use log space: log(D * R_H) = log(A)

                            y_transformed = np.log(d_fit_si * rh_fit_m)  # log(D * R_H)

                            # Use HuberRegressor for robust fitting (no X needed for constant fit)
                            # We're fitting a constant: log(D * R_H) = log(A)
                            regressor = HuberRegressor(fit_intercept=True, alpha=0.0)
                            X_dummy = np.zeros(
                                (len(y_transformed), 1)
                            )  # Dummy variable for constant fit
                            regressor.fit(X_dummy, y_transformed)

                            # Get the fitted constant (log(A))
                            log_A = regressor.intercept_
                            A = np.exp(log_A)
                            exponent = -1.0  # Fixed exponent

                            # Calculate R² using only inliers
                            inlier_mask = ~regressor.outliers_
                            y_inliers = y_transformed[inlier_mask]
                            y_pred_inliers = np.full_like(y_inliers, log_A)
                            r_squared = (
                                r2_score(y_inliers, y_pred_inliers)
                                if len(y_inliers) > 1
                                else 0.0
                            )

                            # Calculate standard errors using only inliers
                            n_inliers = np.sum(inlier_mask)
                            if n_inliers > 1:
                                residuals_inliers = y_inliers - log_A
                                mse_inliers = np.mean(residuals_inliers**2)
                                log_A_std_err = np.sqrt(mse_inliers / n_inliers)
                                A_std_err = A * log_A_std_err
                                exponent_std_err = 0.0  # Fixed, so no uncertainty
                            else:
                                log_A_std_err = 0.0
                                A_std_err = 0.0
                                exponent_std_err = 0.0

                            print(
                                f"Fixed Stokes-Einstein fit results (n={n_inliers} inliers, {np.sum(~inlier_mask)} outliers):"
                            )

                        else:
                            # Variable exponent case: D = A * R_H^n
                            # Log-linear regression: log(D) = log(A) + n * log(R_H)
                            X = np.log(rh_fit_m).reshape(-1, 1)  # log(R_H)
                            y = np.log(d_fit_si)  # log(D)

                            # Fit the linear model in log space
                            regressor = LinearRegression(fit_intercept=True)
                            regressor.fit(X, y)

                            # Get the slope (exponent) and intercept
                            exponent = regressor.coef_[0]  # This is n in D = A * R_H^n
                            log_A = regressor.intercept_  # This is log(A)
                            A = np.exp(log_A)  # Convert back to linear scale

                            # Calculate R² score in log space
                            y_pred_log = regressor.predict(X)
                            r_squared = r2_score(y, y_pred_log)

                            # Calculate standard errors
                            residuals = y - y_pred_log
                            mse = np.mean(residuals**2)
                            X_centered = X - np.mean(X)
                            exponent_variance = mse / np.sum(X_centered**2)
                            exponent_std_err = np.sqrt(exponent_variance)

                            # Standard error for log_A (intercept)
                            n_points = len(X)
                            log_A_variance = mse * (
                                1 / n_points + np.mean(X) ** 2 / np.sum(X_centered**2)
                            )
                            log_A_std_err = np.sqrt(log_A_variance)

                            # Propagate uncertainty to A
                            A_std_err = A * log_A_std_err

                            print(f"Modified Stokes-Einstein fit results:")

                        # Calculate effective viscosity even with modified exponent
                        # For classical Stokes-Einstein: D = k_BT/(6πηR_H), so A = k_BT/(6πη)
                        # Even if exponent ≠ -1, we can still estimate an "effective" viscosity
                        # using the fitted A value at some reference radius
                        if abs(exponent + 1) < 0.1:  # Close to classical exponent
                            eta_fitted = k_B * temperature / (6 * np.pi * A)
                            eta_std_err = (
                                k_B * temperature * A_std_err / (6 * np.pi * A**2)
                            )
                        else:
                            # For non-classical exponent, calculate effective viscosity
                            # at the geometric mean of the radii
                            ref_radius = np.exp(np.mean(np.log(rh_fit_m)))
                            eta_fitted = (
                                k_B
                                * temperature
                                / (6 * np.pi * A * ref_radius ** (exponent + 1))
                            )
                            # Uncertainty propagation is more complex for non-classical case
                            eta_std_err = eta_fitted * np.sqrt(
                                (A_std_err / A) ** 2
                                + (
                                    (exponent + 1)
                                    * exponent_std_err
                                    * np.log(ref_radius)
                                )
                                ** 2
                            )

                        # Generate theoretical curve using fitted parameters
                        rh_range = np.linspace(rh_fit.min(), rh_fit.max(), 100)
                        rh_range_m = rh_range * 1e-10  # Convert to m
                        d_theory_si = A * (rh_range_m**exponent)

                        # Generate theoretical curve using fitted parameters
                        rh_range = np.linspace(rh_fit.min(), rh_fit.max(), 100)
                        rh_range_m = rh_range * 1e-10  # Convert to m
                        d_theory_si = A * (rh_range_m**exponent)

                        if fix_exponent:
                            fit_label = f"Classical S-E fit: $D = A/R_H$, η = {eta_fitted*1000:.2f} ± {eta_std_err*1000:.2f} mPa·s"
                        else:
                            fit_label = f"Modified S-E fit: $D \\propto R_H^{{{exponent:.2f}}}$, η = {eta_fitted*1000:.2f} ± {eta_std_err*1000:.2f} mPa·s"

                        plt.plot(
                            rh_range,
                            d_theory_si,
                            "r-",
                            linewidth=2,
                            label=fit_label,
                        )

                        print(f"  Exponent n: {exponent:.3f} ± {exponent_std_err:.3f}")
                        print(
                            f"  Prefactor A: {A:.2e} ± {A_std_err:.2e} m^({2-exponent})/s"
                        )
                        print(
                            f"  Effective viscosity η: {eta_fitted*1000:.2f} ± {eta_std_err*1000:.2f} mPa·s"
                        )
                        print(f"  η: {eta_fitted:.2e} ± {eta_std_err:.2e} Pa·s")
                        print(f"  Temperature: {temperature:.1f} K")
                        print(f"  R² = {r_squared:.3f}")
                        if not fix_exponent and abs(exponent + 1) > 0.1:
                            ref_radius_ang = np.exp(np.mean(np.log(rh_fit)))
                            print(
                                f"  (Viscosity calculated at reference radius: {ref_radius_ang:.1f} Å)"
                            )

                except Exception as e:
                    print(f"Warning: Could not fit Stokes-Einstein relationship: {e}")

            plt.xlabel("Hydrodynamic Radius (Å)")
            plt.ylabel("Diffusion Coefficient (m²/s)")
            plt.title("Diffusion Coefficient vs Hydrodynamic Radius")
            plt.grid(True, alpha=0.3)
            plt.legend()
            # plt.yscale("log")
            # plt.xscale("log")
            plt.tight_layout()

            if output_dir:
                plt.savefig(
                    f"{output_dir}/diffusion_vs_rh.png", dpi=300, bbox_inches="tight"
                )
            if show_plots:
                plt.show()

        else:
            print(
                "Warning: No overlapping data between diffusion and hydrodynamic radius measurements"
            )

    else:
        print(
            "Warning: Cannot plot D vs R_H without both diffusion and hydrodynamic radius data"
        )


def main():
    """Command-line interface for aggregate lifetime analysis."""
    import argparse

    sns.set_theme(context="talk", style="whitegrid")

    parser = argparse.ArgumentParser(
        description="Analyze aggregate lifetimes in molecular dynamics trajectories"
    )
    parser.add_argument("trajectory", help="Path to trajectory file")
    parser.add_argument("structure", help="Path to structure/topology file")
    parser.add_argument(
        "--cutoff",
        "-c",
        type=float,
        default=4.25,
        help="Distance cutoff for clustering (default: 4.25 Å)",
    )
    parser.add_argument(
        "--min-size",
        "-m",
        type=int,
        default=5,
        help="Minimum cluster size (default: 5 molecules)",
    )
    parser.add_argument(
        "--tail-selection",
        "-t",
        default="name C6 C7 C8 C9 C10 C11 C15 C16 C17 C18 C19 C20",
        help="MDAnalysis selection for tail atoms",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=1,
        help="Step size for trajectory analysis (default: 1)",
    )
    parser.add_argument("--start", type=int, help="Starting frame")
    parser.add_argument("--stop", type=int, help="Ending frame")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )
    parser.add_argument(
        "--load-lifetimes", "-l", help="Load existing lifetime DataFrame from CSV"
    )
    parser.add_argument(
        "--msd",
        "--sd",
        action="store_true",
        dest="msd",
        help="Calculate squared displacement and diffusion coefficients",
    )
    parser.add_argument(
        "--rh",
        "--hydrodynamic-radius",
        action="store_true",
        dest="rh",
        help="Calculate hydrodynamic radius for aggregates by size",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature in Kelvin for effective viscosity calculation (default: 298.15)",
    )
    parser.add_argument(
        "--fix-exponent",
        action="store_true",
        help="Fix Stokes-Einstein exponent to -1 and use HuberRegressor for outlier handling",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate plots for SD analysis"
    )
    parser.add_argument(
        "--plot-fit",
        action="store_true",
        help="Generate SD fit plot for specific aggregate size",
    )
    parser.add_argument(
        "--fit-size",
        type=int,
        default=180,
        help="Aggregate size for SD fit plot (default: 180)",
    )
    parser.add_argument(
        "--plot-dir",
        default=".",
        help="Directory to save plots (default: current directory)",
    )

    args = parser.parse_args()

    # Load or calculate lifetime DataFrame
    if args.load_lifetimes:
        if not args.quiet:
            print(f"Loading lifetime data from: {args.load_lifetimes}")
        results_df = load_lifetime_dataframe(args.load_lifetimes)
    else:
        # Run analysis
        results_df = analyze_aggregate_lifetimes(
            trajectory_path=args.trajectory,
            structure_path=args.structure,
            cutoff=args.cutoff,
            min_cluster_size=args.min_size,
            tail_selection=args.tail_selection,
            step=args.step,
            start=args.start,
            stop=args.stop,
            output_path=args.output,
            verbose=not args.quiet,
        )

    if not args.quiet:
        print(f"\nAnalysis complete!")
        print(f"Found {len(results_df)} unique aggregate lifetimes")
        if len(results_df) > 0:
            print(f"Average lifetime: {results_df['lifetime_time'].mean():.2f} ps")
            print(f"Longest lifetime: {results_df['lifetime_time'].max():.2f} ps")

            # Analyze average lifetime as a function of aggregation number
            print(f"\nAverage lifetime by aggregation number:")
            print("-" * 50)
            lifetime_by_size = results_df.groupby("aggregation_number")[
                "lifetime_time"
            ].agg(["mean", "std", "count"])
            lifetime_by_size = lifetime_by_size.sort_index()

            print(f"{'Size':<6} {'Mean (ps)':<12} {'Std (ps)':<12} {'Count':<8}")
            print("-" * 50)
            for size, row in lifetime_by_size.iterrows():
                mean_val = row["mean"]
                std_val = row["std"] if not pd.isna(row["std"]) else 0.0
                count_val = int(row["count"])
                print(f"{size:<6} {mean_val:<12.2f} {std_val:<12.2f} {count_val:<8}")

            # Additional statistics
            print(f"\nAggregate size statistics:")
            print(
                f"Smallest aggregate: {results_df['aggregation_number'].min()} molecules"
            )
            print(
                f"Largest aggregate: {results_df['aggregation_number'].max()} molecules"
            )
            print(
                f"Most common size: {results_df['aggregation_number'].mode().iloc[0]} molecules"
            )

            # Show distribution of aggregate sizes
            size_counts = results_df["aggregation_number"].value_counts().sort_index()
            print(f"\nAggregate size distribution:")
            print("-" * 30)
            for size, count in size_counts.items():
                percentage = (count / len(results_df)) * 100
                print(f"Size {size:2d}: {count:3d} aggregates ({percentage:5.1f}%)")

    # Initialize variables for optional analyses
    diffusion_df = pd.DataFrame()

    # Run SD analysis if requested
    if args.msd:
        if not args.quiet:
            print(f"\nCalculating squared displacement...")

        sd_df = calculate_aggregate_sd(
            trajectory_path=args.trajectory,
            structure_path=args.structure,
            lifetime_df=results_df,
            tail_selection=args.tail_selection,
            verbose=not args.quiet,
        )

        # Calculate diffusion coefficients
        if not args.quiet:
            print(f"Estimating diffusion coefficients...")

        diffusion_df = estimate_diffusion_coefficients(sd_df, verbose=not args.quiet)

        # Save SD and diffusion data
        if args.output:
            base_name = Path(args.output).stem
            sd_output = f"{base_name}_sd.csv"
            diffusion_output = f"{base_name}_diffusion.csv"

            sd_df.to_csv(sd_output, index=False)
            diffusion_df.to_csv(diffusion_output, index=False)

            if not args.quiet:
                print(f"SD data saved to: {sd_output}")
                print(f"Diffusion data saved to: {diffusion_output}")

        # Print diffusion coefficient results
        if not args.quiet and len(diffusion_df) > 0:
            print(f"\nDiffusion coefficients by aggregate size:")
            print("-" * 70)
            print(
                f"{'Size':<6} {'D (m²/s)':<15} {'Error (m²/s)':<15} {'R²':<8} {'N points':<8}"
            )
            print("-" * 70)
            for _, row in diffusion_df.iterrows():
                # Convert to SI units for display
                d_si = row["diffusion_coefficient"] * 1e-8  # Å²/ps to m²/s
                d_err_si = row["diffusion_error"] * 1e-8  # Å²/ps to m²/s
                print(
                    f"{int(row['aggregation_number']):<6} "
                    f"{d_si:<15.2e} "
                    f"{d_err_si:<15.2e} "
                    f"{row['r_squared']:<8.3f} "
                    f"{int(row['n_points']):<8}"
                )

        # Generate plots if requested
        if args.plot:
            if not args.quiet:
                print(f"\nGenerating plots...")

            plot_sd_analysis(
                sd_df=sd_df,
                diffusion_df=diffusion_df,
                output_dir=args.plot_dir,
                show_plots=not args.quiet,
            )

            if not args.quiet:
                print(f"Plots saved to: {args.plot_dir}")

        # Generate SD fit plot if requested
        if args.plot_fit:
            if not args.quiet:
                print(f"\nGenerating SD fit plot for size {args.fit_size}...")

            plot_sd_fit(
                sd_df=sd_df,
                diffusion_df=diffusion_df,
                aggregate_size=args.fit_size,
                output_dir=args.plot_dir,
                show_plots=not args.quiet,
            )

            if not args.quiet:
                print(f"SD fit plot saved to: {args.plot_dir}")

    # Run hydrodynamic radius analysis if requested
    if args.rh:
        if not args.quiet:
            print(f"\nCalculating hydrodynamic radius...")

        rh_df = calculate_aggregate_hydrodynamic_radius(
            trajectory_path=args.trajectory,
            structure_path=args.structure,
            lifetime_df=results_df,
            tail_selection=args.tail_selection,
            verbose=not args.quiet,
        )

        # Save hydrodynamic radius data
        if args.output:
            base_name = Path(args.output).stem
            rh_output = f"{base_name}_rh.csv"

            rh_df.to_csv(rh_output, index=False)

            if not args.quiet:
                print(f"Hydrodynamic radius data saved to: {rh_output}")

        # Print hydrodynamic radius results
        if not args.quiet and len(rh_df) > 0:
            print(f"\nHydrodynamic radius by aggregate size:")
            print("-" * 80)
            print(f"{'Size':<6} {'R_H (Å)':<12} {'Std (Å)':<12} {'N aggregates':<12}")
            print("-" * 80)
            for _, row in rh_df.iterrows():
                print(
                    f"{int(row['aggregation_number']):<6} "
                    f"{row['hydrodynamic_radius_avg']:<12.2f} "
                    f"{row['hydrodynamic_radius_std']:<12.2f} "
                    f"{int(row['n_aggregates']):<12}"
                )

        # Generate hydrodynamic radius plots if requested
        if args.plot:
            if not args.quiet:
                print(f"\nGenerating hydrodynamic radius plots...")

            # Check if we have diffusion data for combined plotting
            diffusion_df_for_rh = (
                diffusion_df if args.msd and len(diffusion_df) > 0 else None
            )

            plot_hydrodynamic_radius_analysis(
                rh_df=rh_df,
                diffusion_df=diffusion_df_for_rh,
                output_dir=args.plot_dir,
                show_plots=not args.quiet,
                temperature=args.temperature,
                fix_exponent=args.fix_exponent,
            )

            if not args.quiet:
                print(f"Hydrodynamic radius plots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
