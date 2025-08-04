"""Extract a cluster and save it to a new file."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

from aot_analysis.cluster import (
    center_on_cluster,
    count_inner_aot,
    count_inside_vesicle,
    get_adj_array,
    vesicality,
)


def get_clusters(
    universe: mda.Universe, tail_selection: str, cutoff: float
) -> list[tuple[mda.ResidueGroup, int]]:
    """Get all of the clusters in the universe."""
    tailgroups = universe.select_atoms(tail_selection)
    sparse_adj_arr = get_adj_array(tailgroups, cutoff, universe.dimensions)
    n_aggregates, connected_comps = connected_components(sparse_adj_arr, directed=False)

    whole_molecules = tailgroups.residues
    clusters = []
    for i in range(n_aggregates):
        clustered_mols = whole_molecules[np.where(connected_comps == i)]
        agg_num = len(clustered_mols)
        clusters.append((clustered_mols, agg_num))

    return clusters


def center_and_wrap_cluster(universe: mda.Universe, cluster: mda.ResidueGroup):
    """Center the cluster in the middle of the box and wrap all atoms."""
    # Get center of geometry of the cluster
    cog = cluster.atoms.center_of_geometry()

    # Calculate the translation vector to the center of the box
    box_center = universe.dimensions[:3] / 2
    translation = box_center - cog

    # Translate all atoms
    universe.atoms.translate(translation)

    # Wrap all atoms to ensure they're within the primary unit cell
    universe.atoms.wrap()

    return universe


def extract_compact_aot(
    residue_group: mda.ResidueGroup, output_file: str = "compact_aot.pdb"
):
    """Extract the most compact AOT molecule (occupying least volume) and save to PDB."""
    min_volume = float("inf")
    most_compact_residue = None

    # Iterate through all residues to find the one with smallest convex hull volume
    for res in residue_group:
        # Get atom positions
        positions = res.atoms.positions

        # Need at least 4 points to make a 3D convex hull
        if len(positions) < 4:
            continue

        try:
            hull = ConvexHull(positions)
            volume = hull.volume

            if volume < min_volume:
                min_volume = volume
                most_compact_residue = res
        except Exception as e:
            print(f"Could not calculate convex hull for residue {res.resid}: {e}")

    if most_compact_residue is not None:
        # Create a new Universe with just this residue
        temp_u = mda.Merge(most_compact_residue.atoms)
        # Write to PDB file
        temp_u.atoms.write(output_file)
        print(
            f"Most compact AOT molecule (volume: {min_volume:.2f} Å³) saved to {output_file}"
        )
    else:
        print("Could not identify a compact AOT molecule")


def calculate_radii_ranges(residue_group: mda.ResidueGroup, tail_selection: str):
    """Calculate and report ranges of radii for inner and outer shells."""
    # Get tailgroups from the residue group
    tailgroups = residue_group.atoms.select_atoms(tail_selection)
    agg_cog = tailgroups.center_of_geometry()

    # Initialize data containers
    inner_hg_radii = []
    outer_hg_radii = []
    inner_atom_radii = []
    outer_atom_radii = []
    inner_residues = []
    outer_residues = []

    # Identify inner and outer shells
    for res in tailgroups.residues.unique:
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

        # Calculate headgroup radius (distance from center)
        hg_radius = np.linalg.norm(hg_cog - agg_cog)

        # Store based on whether it's inner or outer shell
        if scalar_orientation < 0:  # Inner shell
            inner_hg_radii.append(hg_radius)
            inner_residues.append(res)
        else:  # Outer shell
            outer_hg_radii.append(hg_radius)
            outer_residues.append(res)

    # Calculate radii for all atoms in each shell
    for res in inner_residues:
        for atom in res.atoms:
            radius = np.linalg.norm(atom.position - agg_cog)
            inner_atom_radii.append(radius)

    for res in outer_residues:
        for atom in res.atoms:
            radius = np.linalg.norm(atom.position - agg_cog)
            outer_atom_radii.append(radius)

    # Report the results
    print("\n===== RADII ANALYSIS =====")

    if inner_hg_radii:
        print(
            f"Inner shell headgroups radii range: {min(inner_hg_radii):.2f} - {max(inner_hg_radii):.2f} Å"
        )
    else:
        print("No inner shell headgroups detected")

    if outer_hg_radii:
        print(
            f"Outer shell headgroups radii range: {min(outer_hg_radii):.2f} - {max(outer_hg_radii):.2f} Å"
        )
    else:
        print("No outer shell headgroups detected")

    if inner_atom_radii:
        print(
            f"Inner shell atoms radii range: {min(inner_atom_radii):.2f} - {max(inner_atom_radii):.2f} Å"
        )
    else:
        print("No inner shell atoms detected")

    if outer_atom_radii:
        print(
            f"Outer shell atoms radii range: {min(outer_atom_radii):.2f} - {max(outer_atom_radii):.2f} Å"
        )
    else:
        print("No outer shell atoms detected")

    print("==========================\n")

    return {
        "inner_hg_range": (
            (min(inner_hg_radii), max(inner_hg_radii)) if inner_hg_radii else None
        ),
        "outer_hg_range": (
            (min(outer_hg_radii), max(outer_hg_radii)) if outer_hg_radii else None
        ),
        "inner_atom_range": (
            (min(inner_atom_radii), max(inner_atom_radii)) if inner_atom_radii else None
        ),
        "outer_atom_range": (
            (min(outer_atom_radii), max(outer_atom_radii)) if outer_atom_radii else None
        ),
    }


def analyze_vesicle(
    residue_group: mda.ResidueGroup,
    tail_selection: str,
    vesicality_threshold: float = 0.5,
):
    """Analyze a cluster to determine if it's a vesicle and report its contents."""
    # Get tailgroups from the residue group
    tailgroups = residue_group.atoms.select_atoms(tail_selection)

    # Center the cluster to avoid PBC issues
    center_on_cluster(residue_group.atoms)

    # Calculate vesicality
    vesicality_value = vesicality(tailgroups)

    # If vesicality is below threshold, it's probably not a vesicle
    if vesicality_value < vesicality_threshold:
        print(
            f"This cluster is likely not a vesicle (vesicality = {vesicality_value:.3f})"
        )
        return False

    # Count inner layer AOT molecules
    inner_aot_count = count_inner_aot(tailgroups)

    # Count water and counterions inside
    water_count = count_inside_vesicle(residue_group, tailgroups, "W")
    counterion_count = count_inside_vesicle(residue_group, tailgroups, "NA")

    # Calculate outer layer AOT molecules
    total_aot = len(residue_group)
    outer_aot_count = total_aot - inner_aot_count

    print("\n===== VESICLE ANALYSIS =====")
    print(f"Vesicality: {vesicality_value:.3f}")
    print(f"Inner layer AOT molecules: {inner_aot_count}")
    print(f"Outer layer AOT molecules: {outer_aot_count}")
    print(f"Water molecules inside: {water_count}")
    print(f"Counterions inside: {counterion_count}")
    print("============================\n")

    # Calculate radii ranges
    calculate_radii_ranges(residue_group, tail_selection)

    return True


def plot_vesicle_3d(
    residue_group: mda.ResidueGroup,
    tail_selection: str,
    output_file: str = None,
):
    """Create a 3D plot showing AOT orientations and sodium ions inside the vesicle."""
    # Get tailgroups from the residue group
    tailgroups = residue_group.atoms.select_atoms(tail_selection)

    # Calculate cluster center of geometry
    agg_cog = tailgroups.center_of_geometry()

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Prepare data for plotting
    tail_cogs = []  # Tail group centers of geometry
    head_cogs = []  # Head group centers of geometry
    orientations = []  # Vectors from tail CoG to head CoG
    inner_layer = []  # Boolean flags for inner layer molecules

    # Calculate orientations for all AOT molecules
    for res in tailgroups.residues.unique:
        headgroup = res.atoms.difference(tailgroups)
        tailgroup = res.atoms.intersection(tailgroups)

        hg_cog = headgroup.center_of_geometry()
        tail_cog = tailgroup.center_of_geometry()

        # Get the orientation vector
        orientation = hg_cog - tail_cog
        orientation_normalized = orientation / np.linalg.norm(orientation)

        from_agg_centre = hg_cog - agg_cog
        dist_from_centre = np.linalg.norm(from_agg_centre)

        from_agg_centre_normalized = from_agg_centre / dist_from_centre
        scalar_orientation = np.dot(orientation_normalized, from_agg_centre_normalized)

        # Store data for plotting
        tail_cogs.append(tail_cog)
        head_cogs.append(hg_cog)
        orientations.append(orientation)
        inner_layer.append(
            scalar_orientation < 0
        )  # True if it's an inner layer molecule

    # Convert to numpy arrays for easier manipulation
    tail_cogs = np.array(tail_cogs)
    head_cogs = np.array(head_cogs)
    orientations = np.array(orientations)
    inner_layer = np.array(inner_layer)

    # Get sodium atoms inside the vesicle
    universe = residue_group.universe
    na_atoms = universe.select_atoms("resname NA")

    # Calculate shell radius similar to count_inside_vesicle
    outer_hg_radii = []
    inner_hg_radii = []
    outer_weights = []
    inner_weights = []

    for i, res in enumerate(tailgroups.residues.unique):
        headgroup = res.atoms.difference(tailgroups)
        tailgroup = res.atoms.intersection(tailgroups)

        hg_cog = headgroup.center_of_geometry()
        from_agg_centre = hg_cog - agg_cog
        dist_from_centre = np.linalg.norm(from_agg_centre)

        if inner_layer[i]:
            inner_hg_radii.append(dist_from_centre)
            inner_weights.append(1.0)
        else:
            outer_hg_radii.append(dist_from_centre)
            outer_weights.append(1.0)

    # If there's no inner layer, return
    if not inner_hg_radii:
        print("No inner layer detected, skipping 3D plot")
        return

    # Get average inner radius
    r_inner = np.average(inner_hg_radii, weights=inner_weights)

    # Find sodium atoms inside the vesicle
    na_inside_positions = []
    box = universe.dimensions
    for atom in na_atoms:
        diff = atom.position - agg_cog
        # Apply minimum image convention
        for i in range(3):
            if diff[i] > box[i] / 2:
                diff[i] -= box[i]
            elif diff[i] < -box[i] / 2:
                diff[i] += box[i]
        distance = np.linalg.norm(diff)
        if distance < r_inner:
            na_inside_positions.append(atom.position)

    # Plot AOT molecules as arrows
    for i in range(len(tail_cogs)):
        color = "red" if inner_layer[i] else "blue"
        # Plot arrow showing AOT orientation
        ax.quiver(
            tail_cogs[i, 0],
            tail_cogs[i, 1],
            tail_cogs[i, 2],  # Start point
            orientations[i, 0],
            orientations[i, 1],
            orientations[i, 2],  # Direction
            color=color,
            alpha=0.6,
            arrow_length_ratio=0.2,
            length=1.0,  # Scale all arrows to same length for clarity
            normalize=True,
        )

    # Plot sodium ions inside vesicle
    if na_inside_positions:
        na_inside_positions = np.array(na_inside_positions)
        ax.scatter(
            na_inside_positions[:, 0],
            na_inside_positions[:, 1],
            na_inside_positions[:, 2],
            c="green",
            marker="o",
            s=50,
            label="Sodium ions inside",
        )

    # Create a ghost point for inner and outer layer in the legend
    ax.scatter([], [], [], c="red", marker=">", s=100, label="Inner layer AOT")
    ax.scatter([], [], [], c="blue", marker=">", s=100, label="Outer layer AOT")

    # Add a wireframe sphere to represent the inner layer boundary
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = r_inner * np.outer(np.cos(u), np.sin(v)) + agg_cog[0]
    y = r_inner * np.outer(np.sin(u), np.sin(v)) + agg_cog[1]
    z = r_inner * np.outer(np.ones(np.size(u)), np.cos(v)) + agg_cog[2]
    ax.plot_surface(x, y, z, alpha=0.1, color="gray")

    # Set labels and title
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title("Vesicle Structure Visualization")

    # Set equal aspect ratio for all axes
    max_range = (
        np.array(
            [
                tail_cogs[:, 0].max() - tail_cogs[:, 0].min(),
                tail_cogs[:, 1].max() - tail_cogs[:, 1].min(),
                tail_cogs[:, 2].max() - tail_cogs[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (tail_cogs[:, 0].max() + tail_cogs[:, 0].min()) * 0.5
    mid_y = (tail_cogs[:, 1].max() + tail_cogs[:, 1].min()) * 0.5
    mid_z = (tail_cogs[:, 2].max() + tail_cogs[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add legend
    ax.legend()

    # Tight layout
    plt.tight_layout()

    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"3D visualization saved to {output_file}")

    # Show plot
    plt.show()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("struct", help="Structure file.")
    parser.add_argument("tails", type=str, help="Tailgroup selection string.")
    parser.add_argument(
        "-c", "--cutoff", type=float, default=0.5, help="Cutoff for clustering."
    )
    parser.add_argument("-o", default="aggregate.gro", help="Output file.")
    parser.add_argument(
        "--min-size",
        type=int,
        help="Extract all clusters above this size. The index of each cluster will be appended to the end of the output file stem.",
    )
    parser.add_argument("-i", type=int, help="Index of the cluster.")
    parser.add_argument(
        "--vesicle-threshold",
        type=float,
        default=0.5,
        help="Vesicality threshold for identifying vesicles (default: 0.5).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze clusters, don't extract them.",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Generate a 3D plot showing AOT orientations and sodium ions inside the vesicle.",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Save the 3D plot to this file (requires --plot-3d).",
    )
    parser.add_argument(
        "--extract-compact",
        action="store_true",
        help="Extract the most compact AOT molecule (occupying least volume).",
    )
    parser.add_argument(
        "--compact-file",
        type=str,
        default="compact_aot.pdb",
        help="File to save the most compact AOT molecule (requires --extract-compact).",
    )
    args = parser.parse_args()

    output_file = Path(args.o)

    gro_file = Path(args.struct)
    if not gro_file.exists():
        raise FileNotFoundError(f"{gro_file} does not exist.")

    print("Clustering...")

    univ = mda.Universe(gro_file)
    clusters = get_clusters(univ, args.tails, args.cutoff)

    if args.i is None:
        for i, (cluster, size) in enumerate(clusters):
            if args.min_size is None:
                print(f"Cluster {i}: {size} molecules")
                continue
            elif size >= args.min_size:
                # Center and wrap the cluster before analysis
                univ = center_and_wrap_cluster(univ, cluster)

                # Analyze the vesicle properties
                print(f"\nAnalyzing cluster {i} with {size} molecules:")
                is_vesicle = analyze_vesicle(
                    cluster, args.tails, args.vesicle_threshold
                )

                # Extract most compact AOT if requested
                if args.extract_compact:
                    compact_file = f"{args.compact_file.split('.')[0]}_{i}.{args.compact_file.split('.')[1]}"
                    extract_compact_aot(cluster, compact_file)

                # Generate 3D plot if requested
                if args.plot_3d:
                    plot_file = args.plot_file or f"vesicle_plot_{i}.png"
                    plot_vesicle_3d(cluster, args.tails, plot_file)

                if not args.analyze_only:
                    out_file = output_file.stem + f"_{i}" + output_file.suffix
                    cluster.atoms.write(out_file)
                    print(f"Cluster {i}, size {size} written to {out_file}")

        if args.min_size is None:
            cluster_idx = int(
                input("Enter the index of the cluster you want to extract: ")
            )
        else:
            # We've already written them all to disk
            return

    else:
        cluster_idx = int(args.i)

    cluster, size = clusters[cluster_idx]

    # Center and wrap the cluster before analysis
    univ = center_and_wrap_cluster(univ, cluster)

    # Analyze the vesicle properties for the selected cluster
    print(f"\nAnalyzing cluster {cluster_idx} with {size} molecules:")
    is_vesicle = analyze_vesicle(cluster, args.tails, args.vesicle_threshold)

    # Extract most compact AOT if requested
    if args.extract_compact:
        extract_compact_aot(cluster, args.compact_file)

    # Generate 3D plot if requested
    if args.plot_3d:
        plot_file = args.plot_file or f"vesicle_plot_{cluster_idx}.png"
        plot_vesicle_3d(cluster, args.tails, plot_file)

    if not args.analyze_only:
        cluster.atoms.write(args.o)
        print(f"Cluster {cluster_idx} written to {args.o}")


if __name__ == "__main__":
    main()
