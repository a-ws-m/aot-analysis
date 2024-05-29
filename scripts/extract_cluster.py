"""Extract a cluster and save it to a new file."""

from argparse import ArgumentParser
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from scipy.sparse.csgraph import connected_components

from aot.cluster import get_adj_array


def get_clusters(
    universe: mda.Universe, tail_selection: str, cutoff: float
) -> list[tuple[mda.ResidueGroup, int]]:
    """Get all of the clusters in the universe."""
    tailgroups = univ.seelct_atoms(tail_selection)
    sparse_adj_arr = get_adj_array(tailgroups, cutoff, universe.dimensions)
    _, connected_comps = connected_components(sparse_adj_arr, directed=False)

    _, agg_indices, agg_nums = np.unique(
        connected_comps, return_counts=True, return_index=True
    )
    # agg_indices contains the index of the first molecule in each aggregate

    whole_molecules = tailgroups.residues
    clusters = []
    for i, agg_num in zip(range(len(agg_indices)), agg_nums):
        start_idx = agg_indices[i]
        try:
            end_idx = agg_indices[i + 1]
            agg_residues = whole_molecules[start_idx:end_idx]
        except IndexError:
            agg_residues = whole_molecules[start_idx:]

        clusters.append((agg_residues, agg_num))

    return clusters


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("struct", required=True, help="Structure file.")
    parser.add_argument(
        "tails", type=str, required=True, help="Tailgroup selection string."
    )
    parser.add_argument(
        "cutoff", type=float, default=0.5, help="Cutoff for clustering."
    )
    parser.add_argument("-o", default="output.gro", help="Output file.")
    parser.add_argument("-i", type=int, help="Index of the cluster.")
    args = parser.parse_args()

    gro_file = Path(args.struct)
    if not gro_file.exists():
        raise FileNotFoundError(f"{gro_file} does not exist.")

    print("Clustering...")

    univ = mda.Universe(gro_file)
    clusters = get_clusters(univ, args.tails, args.cutoff)

    if args.i is None:
        for i, (cluster, size) in enumerate(clusters):
            print(f"Cluster {i}: {size} molecules")

        cluster_idx = int(input("Enter the index of the cluster you want to extract: "))
    else:
        cluster_idx = int(args.i)

    cluster, _ = clusters[cluster_idx]
    cluster.atoms.write(args.o)
    print(f"Cluster {cluster_idx} written to {args.o}")


if __name__ == "__main__":
    main()
