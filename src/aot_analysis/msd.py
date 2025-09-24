"""
Script to compute mean squared displacement (MSD) of aggregates of various sizes from a trajectory and adjacency matrix file.

Usage:
    python msd.py <tpr_file> <traj_file> <adj_file>

Outputs:
    Prints MSD as a function of time for each aggregate size.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse.csgraph import connected_components
from scipy.stats import linregress
from tqdm import tqdm

from aot_analysis.utilities import load_sparse


def get_aggregates(adj_mat):
    """Return a list of sets, each set is the indices of molecules in an aggregate."""

    n_components, labels = connected_components(adj_mat, directed=False)
    aggregates = defaultdict(list)
    for idx, label in enumerate(labels):
        aggregates[label].append(idx)
    return [set(members) for members in aggregates.values()]


def compute_aggregate_com(universe, aggregate_atom_indices):
    """Compute center of mass for a set of atom indices."""
    ag = universe.atoms[aggregate_atom_indices]
    return ag.center_of_mass()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate aggregate MSD and diffusivity from trajectory and adjacency matrix."
    )
    parser.add_argument("tpr_file", type=str, help="Path to topology file (tpr)")
    parser.add_argument("traj_file", type=str, help="Path to trajectory file")
    parser.add_argument(
        "adj_file", type=str, help="Path to adjacency matrix file (npz)"
    )
    args = parser.parse_args()
    tpr_file = Path(args.tpr_file)
    traj_file = Path(args.traj_file)
    adj_file = Path(args.adj_file)

    # Load trajectory
    u = mda.Universe(tpr_file, traj_file)
    # Determine dt (time between frames) in picoseconds
    dt_ps = u.trajectory.dt  # dt in ps
    # Load adjacency matrices
    adj_mats = load_sparse(adj_file)
    # Map: (aggregate_size) -> list of arrays (each array: trajectory of COMs for a unique aggregate)
    aggregate_trajs = defaultdict(list)
    # For each aggregate size, keep track of currently active aggregates in previous frame
    prev_aggregates = dict()  # agg_size -> list of (agg_tuple, traj_idx)
    for frame_idx, adj_mat in tqdm(
        sorted(adj_mats.items()), desc="Computing aggregate trajectories"
    ):
        u.trajectory[frame_idx]
        aggregates = get_aggregates(adj_mat)
        curr_agg_map = dict()  # agg_size -> list of (agg_tuple, traj_idx)
        # For each aggregate, track its members and COM
        for agg in aggregates:
            agg_size = len(agg)
            agg_tuple = tuple(sorted(agg))
            # Check if this aggregate is a direct continuation from previous frame
            prev_list = prev_aggregates.get(agg_size, [])
            found = False
            for i, (prev_tuple, traj_idx) in enumerate(prev_list):
                if prev_tuple == agg_tuple:
                    # Continue previous trajectory
                    com = compute_aggregate_com(u, list(agg))
                    aggregate_trajs[agg_size][traj_idx].append((frame_idx, com))
                    curr_agg_map.setdefault(agg_size, []).append((agg_tuple, traj_idx))
                    found = True
                    break
            if not found:
                # Start new trajectory for this aggregate
                com = compute_aggregate_com(u, list(agg))
                aggregate_trajs[agg_size].append([(frame_idx, com)])
                new_traj_idx = len(aggregate_trajs[agg_size]) - 1
                curr_agg_map.setdefault(agg_size, []).append((agg_tuple, new_traj_idx))
        prev_aggregates = curr_agg_map
    # Now, for each aggregate size, compute MSD

    results_list = []
    print("Aggregate size | Time (frames) | MSD (A^2)")
    for agg_size, trajs in aggregate_trajs.items():
        msd_by_dt = defaultdict(list)
        for traj in trajs:
            if len(traj) < 2:
                continue
            frames, coms = zip(*traj)
            coms = np.array(coms)
            for dt in range(1, len(coms)):
                displacements = coms[dt:] - coms[:-dt]
                sq_disp = np.sum(displacements**2, axis=1)
                msd_by_dt[dt].extend(sq_disp)
        # Prepare arrays for regression
        dts = []
        msd_means = []
        msd_stds = []
        for dt, msds in sorted(msd_by_dt.items()):
            mean_msd = np.mean(msds)
            std_msd = np.std(msds)
            dts.append(dt)
            msd_means.append(mean_msd)
            msd_stds.append(std_msd)
            print(f"{agg_size:13d} | {dt:12d} | {mean_msd:10.3f}")
        # Linear regression for diffusivity (Einstein relation: MSD = 6Dt)
        if len(dts) > 1:
            dts_arr = np.array(dts) * dt_ps / 1000.0  # convert to nanoseconds
            msd_means_arr = np.array(msd_means)
            result = linregress(dts_arr, msd_means_arr)
            # If result is a tuple, unpack; if LinregressResult, use attributes
            try:
                slope = result.slope  # type:ignore
                std_err = result.stderr  # type:ignore
                r_value = result.rvalue  # type:ignore
            except AttributeError:
                slope, intercept, r_value, p_value, std_err = result
            D = slope / 6.0  # type:ignore
            D_err = std_err / 6.0  # type:ignore
            r2 = r_value**2  # type:ignore
            print(
                f"Aggregate size {agg_size}: D = {D:.5e} +/- {D_err:.2e} (A^2/ns), R^2 = {r2:.3f}"
            )
            results_list.append((agg_size, D, D_err))
    # Save results to file

    df = pd.DataFrame(
        results_list, columns=["Aggregate size", "Diffusivity (A^2/ns)", "Std error"]
    )
    df.to_csv("diffusivities.csv", index=False)
    # Plot diffusivity vs aggregate size
    if not df.empty:
        sns.set_theme(style="whitegrid")
        plt.errorbar(
            df["Aggregate size"],
            df["Diffusivity (A^2/ns)"],
            yerr=df["Std error"],
            fmt="o",
            capsize=4,
            label="Diffusivity",
        )
        plt.xlabel("Aggregate size")
        plt.ylabel(r"Diffusivity ($\mathrm{\AA}^2$/ns)")
        plt.title("Diffusivity vs Aggregate Size")
        plt.legend()
        plt.tight_layout()
        plt.savefig("diffusivity_vs_size.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
