import argparse
from functools import lru_cache
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis.distances import self_distance_array
from tqdm import tqdm

from .cluster import batch_ma_analysis
from .utilities import *

TIME_COL = r"Time ($\mathrm{\mu s}$)"


@lru_cache
def load_results_datasets(
    results: tuple[AtomisticResults | CoarseResults, ...],
    min_cluster_size: int = 5,
    dir_: Path = Path("."),
    end: Optional[int] = None,
    end_time: Optional[int] = None,
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

        this_df = this_df[this_df["Frame"] <= end] if end is not None else this_df
        this_df = this_df[
            this_df[AggregateProperties.AGGREGATION_NUMBERS.value] >= min_cluster_size
        ]
        # this_df = this_df.groupby("Time (ps)").mean()
        # this_df["Time (ps)"] = this_df.index
        this_df["% AOT"] = result.percent_aot
        this_df["Simulation"] = result.plot_name
        this_df["Type"] = (
            result.counterion.longname
            if isinstance(result, AtomisticResults)
            else result.coarseness.friendly_name
        )

        plot_df = pd.concat([plot_df, this_df], ignore_index=True)

    if end_time is not None:
        plot_df = plot_df[plot_df["Time (ps)"] <= end_time]

    plot_df[TIME_COL] = plot_df["Time (ps)"] * 1e-6
    plot_df["Log agg. num"] = plot_df[
        AggregateProperties.AGGREGATION_NUMBERS.value
    ].apply(np.log10)
    plot_df["Norm. agg. number"] = plot_df[
        AggregateProperties.NORMALISED_AGGREGATION_NUMBERS.value
    ]

    print(plot_df.head())

    try:
        plot_df[AggregateProperties.SURFACE_AREA_PER_SURFACTANT.value] = (
            plot_df[AggregateProperties.SURFACE_AREA.value]
            / plot_df[AggregateProperties.AGGREGATION_NUMBERS.value]
        )
        plot_df[AggregateProperties.SURFACE_AREA_TO_VOLUME.value] = (
            plot_df[AggregateProperties.SURFACE_AREA.value]
            / plot_df[AggregateProperties.VOLUME.value]
        )
        plot_df[AggregateProperties.VOL_PER_SURFACTANT.value] = (
            plot_df[AggregateProperties.VOLUME.value]
            / plot_df[AggregateProperties.AGGREGATION_NUMBERS.value]
        )
    except KeyError:
        pass

    return plot_df


def compare_val(
    results: list[AtomisticResults | CoarseResults],
    graph_file: Path,
    y_axis: str,
    min_cluster_size: int = 5,
    end: Optional[int] = None,
    end_time: Optional[int] = None,
    round_: Optional[int] = None,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = load_results_datasets(
        tuple(results), min_cluster_size, end=end, end_time=end_time
    )

    if round_ is not None:
        plot_df[TIME_COL] = plot_df[TIME_COL].round(round_)

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
        errorbar="ci",
        # margin_titles=True,
        # sharey="row",
        facet_kws={"margin_titles": True, "despine": False, "sharey": "row"},
    )
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


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
    hue_norm: Optional[str] = None,
    rename: Optional[str] = None,
    marker: str = ".",
    end: Optional[int] = None,
    end_time: Optional[int] = None,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = load_results_datasets(
        tuple(results), min_cluster_size, end=end, end_time=end_time
    )
    # print(plot_df.describe())

    if use_interval:
        plot_df = plot_df[plot_df["Frame"] % interval == 0]

    if rename is not None:
        plot_df[rename] = plot_df[y_axis]

    print("Done analysing results!")
    print("Plotting graphs.")

    plot_df["Model"] = plot_df["Type"]

    g = sns.catplot(
        data=plot_df,
        x=TIME_COL,
        # order=time_labels,
        y=y_axis if not rename else rename,
        row="% AOT",
        col="Model",
        hue=hue,
        hue_norm=hue_norm,
        native_scale=True,
        kind="strip",
        # inner=None,
        sharey="row",
        sharex="row",
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

    g.set_titles(col_template="{col_name}")

    # g.set_xticklabels(time_labels, rotation=45)
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def compare_soap_similarity(
    results: list[AtomisticResults | CoarseResults],
    file_template: str = "{conc}-soap-similarity-{hue}.pdf",
    use_interval: bool = False,
    interval: int = 50,
    min_cluster_size: int = 5,
    end: Optional[int] = None,
    end_time: Optional[int] = None,
):
    """Plot the KPCA map of the SOAP vectors."""
    for conc in set(result.percent_aot for result in results):
        conc_results = [result for result in results if result.percent_aot == conc]
        plot_df = load_results_datasets(
            tuple(conc_results), min_cluster_size, end=end, end_time=end_time
        )

        if use_interval:
            plot_df = plot_df[plot_df["Frame"] % interval == 0]

        print("Done analysing results!")
        print("Plotting graphs.")

        for hue, hue_name in [("Norm. agg. number", "agg-num"), (TIME_COL, "time")]:

            with sns.axes_style("white"):
                g = sns.relplot(
                    kind="scatter",
                    data=plot_df,
                    x=AggregateProperties.SOAP_SIM_1.value,
                    y=AggregateProperties.SOAP_SIM_2.value,
                    row="% AOT",
                    col="Type",
                    hue=hue,
                    facet_kws={
                        "margin_titles": True,
                        "despine": False,
                        "sharex": False,
                        "sharey": False,
                    },
                    palette="flare",
                )

            g.set_titles(col_template="{col_name}")

            g.tight_layout()
            g.savefig(file_template.format(conc=conc, hue=hue_name), transparent=False)


def compare_cpe(
    results: list[AtomisticResults | CoarseResults],
    graph_file: Path,
    use_interval: bool = False,
    interval: int = 50,
    min_cluster_size: int = 5,
    end: Optional[int] = None,
    end_time: Optional[int] = None,
):
    """Compare the clustering behaviour of several simulations."""
    plot_df = load_results_datasets(
        tuple(results), min_cluster_size, end=end, end_time=end_time
    )

    if use_interval:
        plot_df = plot_df[plot_df["Frame"] % interval == 0]

    print("Done analysing results!")
    print("Plotting graphs.")

    g = sns.relplot(
        kind="scatter",
        data=plot_df,
        x=AggregateProperties.EAB.value,
        y=AggregateProperties.EAC.value,
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

    g.set_titles(col_template="{col_name}")

    g.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def compare_clustering(
    results: "list[AtomisticResults | CoarseResults]",
    graph_file: Path,
    min_cluster_size: int = 5,
    dir_: Path = Path("."),
    end: Optional[int] = None,
    end_time: Optional[int] = None,
):
    """Compare the clustering behaviour of several simulations."""

    plot_df = load_results_datasets(
        tuple(results), min_cluster_size, dir_, end=end, end_time=end_time
    )

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
        errorbar="ci",
        # margin_titles=True,
        # sharey="row",
        palette="colorblind",
        hue_order=["Finest", "Mixed", "Coarsest"],
        facet_kws={"margin_titles": True, "despine": False, "sharey": "row"},
    )

    g.figure.suptitle(
        f"Clustering comparison with min cluster size = {min_cluster_size}"
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels(TIME_COL, "Value")
    g.tight_layout()
    g.savefig(graph_file)


def plot_concentrations(
    results: "list[AtomisticResults | CoarseResults]",
    properties: set[AggregateProperties],
    min_cluster_size: int = 5,
    file_template: str = "{conc}-overview.pdf",
    end: Optional[int] = None,
    end_time: Optional[int] = None,
    round_: Optional[int] = 1,
):
    """Make one plot per concentration showing the evolution of the properties."""
    for conc in set(result.percent_aot for result in results):
        conc_results = [result for result in results if result.percent_aot == conc]
        plot_df = load_results_datasets(
            tuple(conc_results), min_cluster_size, end=end, end_time=end_time
        )

        if round_:
            plot_df[TIME_COL] = plot_df[TIME_COL].round(round_)

        plot_df = plot_df.melt(
            [TIME_COL, "Type"],
            value_vars=[prop.value for prop in properties],
            var_name="Property",
            value_name="Value",
        )

        plot_df["Model"] = plot_df["Type"]

        g = sns.relplot(
            plot_df,
            x=TIME_COL,
            y="Value",
            col="Property",
            col_wrap=3,
            # kind="scatter",
            errorbar="ci",
            hue="Model",
            hue_order=["Finest", "Mixed", "Coarsest"],
            palette="colorblind",
            kind="line",
            # legend=False,
            facet_kws={
                "sharey": False,
                "margin_titles": False,
                "despine": False,
            },
        )
        g.set_titles("")
        for ax, prop in zip(g.axes.flatten(), properties):
            ax.set_ylabel(prop.value)

        conc_str = (
            f"{conc:.0f}" if np.isclose(conc, np.round(conc, 0)) else f"{conc:.2f}"
        )

        # g.figure.subplots_adjust(top=0.99)
        # g.figure.suptitle(f"{conc_str} wt.% AOT")

        g.tight_layout()
        g.savefig(file_template.format(conc=conc), transparent=False)


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
            {r"Distance ($\mathrm{\AA}$)": rdf.results.bins, r"$g(r)$": rdf.results.rdf}
        )
        df["Mapping"] = result.coarseness.friendly_name
        df["% AOT"] = str(result.percent_aot)
        plot_df = pd.concat([plot_df, df], ignore_index=True)

    g = sns.relplot(
        data=plot_df,
        x=r"Distance ($\mathrm{\AA}$)",
        y=r"$g(r)$",
        col="Mapping",
        kind="line",
        hue="% AOT",
        # margin_titles=True,
        facet_kws={"margin_titles": True, "despine": False},
    )
    g.tight_layout()
    g.savefig(graph_file, transparent=False)


def plot_coordnum(
    results: "list[CoarseResults]",
    file_template: str = "coordnum-{percent}.pdf",
    step: int = 1000,
    start: int = 0,
    end: Optional[int] = None,
):
    """Plot the coordination numbers at different radial cutoffs for a given number of timesteps.

    This function calculates the total coordination number between tailgroup
    beads at `num` uniform intervals. The results are plotted as a function of
    the radial cutoff. Each model is on the same plot, but there is one plot per
    timestep. Each timestep subplot is given a title indicating the number of
    aggregates at that timestep, and the average aggregation number.

    If there's only one mapping in the results, all distributions will be plotted
    on a single axis with time as the hue.
    """
    data = {
        r"Distances ($\mathrm{\AA}$)": [],
        "Time (ns)": [],
        "Mapping": [],
        "% AOT": [],
    }

    for result in results:
        u = result.universe()
        tail_atoms = u.select_atoms(result.tail_match)

        num_combinations = len(tail_atoms) * (len(tail_atoms) - 1) // 2
        dists = np.empty((num_combinations), dtype=float)

        # Handle slice with optional end parameter
        trajectory_slice = u.trajectory[start::step]
        if end is not None and end > 0:
            trajectory_slice = u.trajectory[start:end:step]

        for ts in tqdm(
            trajectory_slice,
            desc=f"Calculating RDFs for {result.plot_name}",
        ):
            # Calculate the RDF
            self_distance_array(tail_atoms, box=u.dimensions, result=dists)

            new_data = pd.DataFrame(
                {
                    r"Distances ($\mathrm{\AA}$)": dists,
                    "Time (ns)": [ts.time * 1e-3] * num_combinations,
                    "Mapping": [result.coarseness.friendly_name] * num_combinations,
                    "% AOT": [str(result.percent_aot)] * num_combinations,
                }
            )
            for key in data.keys():
                data[key] += new_data[key].tolist()

    df = pd.DataFrame(data)

    for percent_aot in df["% AOT"].unique():
        plot_df = df[df["% AOT"] == percent_aot]

        # Check if there's only one mapping
        unique_mappings = plot_df["Mapping"].unique()
        single_mapping = len(unique_mappings) == 1

        if single_mapping:
            # Plot all distributions on a single axis with time as the hue
            g = sns.displot(
                data=plot_df,
                x=r"Distances ($\mathrm{\AA}$)",
                hue="Time (ns)",
                kind="ecdf",
                palette="viridis",
                facet_kws={"despine": False},
                alpha=0.7,
            )
            g.set_titles(f"{percent_aot}% AOT - {unique_mappings[0]}")
        else:
            # Original behavior: plot with mapping as hue and time as columns
            g = sns.displot(
                data=plot_df,
                x=r"Distances ($\mathrm{\AA}$)",
                col="Time (ns)",
                col_wrap=3,
                hue="Mapping",
                kind="ecdf",
                facet_kws={"margin_titles": True, "despine": False},
            )

        g.tight_layout()
        g.savefig(file_template.format(percent=percent_aot), transparent=False)


def plot_vesicle_contents(
    results: list[AtomisticResults | CoarseResults],
    graph_file: Path,
    use_interval: bool = False,
    interval: int = 50,
    min_cluster_size: int = 5,
    end: Optional[int] = None,
    end_time: Optional[int] = None,
):
    """Plot the contents inside vesicles (counterions, water, inner AOT molecules)."""
    plot_df = load_results_datasets(
        tuple(results), min_cluster_size, end=end, end_time=end_time
    )

    if use_interval:
        plot_df = plot_df[plot_df["Frame"] % interval == 0]

    # Filter for structures with vesicality > 0.5 (reasonably vesicle-like)
    plot_df = plot_df[plot_df[AggregateProperties.VESICALITY.value] > 0.5]

    # Melt the dataframe to get the contents data in long format
    content_cols = [
        AggregateProperties.COUNTERIONS_INSIDE.value,
        AggregateProperties.WATER_INSIDE.value,
        AggregateProperties.INNER_AOT.value,
    ]

    plot_dfm = plot_df.melt(
        id_vars=["% AOT", "Type", TIME_COL, AggregateProperties.VESICALITY.value],
        value_vars=content_cols,
        var_name="Content Type",
        value_name="Count",
    )

    print("Plotting vesicle contents...")

    # Create the plot
    g = sns.relplot(
        data=plot_dfm,
        x=TIME_COL,
        y="Count",
        col="% AOT",
        hue="Content Type",
        row="Type",
        kind="line",
        errorbar="ci",
        facet_kws={"margin_titles": True, "despine": False},
    )

    g.set_titles(col_template="{col_name}% AOT", row_template="{row_name}")
    g.tight_layout()
    g.savefig(graph_file, transparent=False)

    # Create a scatter plot of vesicality vs. content
    h = sns.relplot(
        data=plot_dfm,
        x=AggregateProperties.VESICALITY.value,
        y="Count",
        col="Content Type",
        hue="Type",
        row="% AOT",
        kind="scatter",
        alpha=0.7,
        facet_kws={"margin_titles": True, "despine": False},
    )

    h.set_titles(row_template="{row_name}% AOT", col_template="{col_name}")
    h.tight_layout()
    h.savefig(str(graph_file).replace(".pdf", "-vs-vesicality.pdf"), transparent=False)


def main():
    """Commandline interface for program."""
    sns.set_theme(context="paper", palette="colorblind")

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
        "-e",
        "--end",
        type=int,
        default=-1,
        help="End the analysis at this frame.",
    )
    parser.add_argument(
        "-m",
        "--min-size",
        default=5,
        type=int,
        help="Minimum cluster size to consider. WARNING: You must overwrite or use `--no-calc` if you change this.",
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=-1,
        help="Plot until this timestep (ps). Negative values mean no limit.",
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
    parser.add_argument(
        "--no-calc",
        action="store_true",
        help="Force the program to skip the calculation step.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to use for the calculations. Default is 1 (no parallelisation).",
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
        "--one-per-conc",
        action="store_true",
        help="Make one plot per concentration showing the evolution of the properties.",
    )
    plot_options.add_argument(
        "--total-vol",
        action="store_true",
        help="Compute the volume estimation from the JAX-enabled Gaussian KDE.",
    )
    plot_options.add_argument(
        "--soap-similarity",
        action="store_true",
        help="Compute the KPCA reduction of the SOAP vectors for each aggregate.",
    )
    plot_options.add_argument(
        "--coordnum",
        action="store_true",
        help="Plot the coordination numbers at different radial cutoffs.",
    )
    plot_options.add_argument(
        "--vesicle-contents",
        action="store_true",
        help="Plot the number of counterions, water molecules, and inner AOT in vesicles.",
    )
    plot_options.add_argument(
        "--disable-interval",
        action="store_true",
        help="Disable the use of intervals when plotting. This will plot every frame.",
    )

    args = parser.parse_args()

    end = args.end if args.end > 0 else None
    end_time = args.end_time if args.end_time > 0 else None
    if end is not None and end_time is not None:
        raise RuntimeError("Cannot specify both --end and --end-time.")

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

    if args.coordnum:
        plot_coordnum(
            results,
            start=args.start,
            step=args.step_size,
            end=end,
        )
        return

    properties = AggregateProperties.fast()
    if args.vol or args.surf or args.sa_ratio:
        properties |= {AggregateProperties.VOLUME, AggregateProperties.SURFACE_AREA}

    if args.total_vol:
        properties |= {AggregateProperties.TOTAL_VOLUME}

    if args.vesicle_contents:
        properties |= {
            AggregateProperties.VESICALITY,
            AggregateProperties.COUNTERIONS_INSIDE,
            AggregateProperties.WATER_INSIDE,
            AggregateProperties.INNER_AOT,
        }

    if not args.no_calc:
        batch_ma_analysis(
            results,
            min_cluster_size=args.min_size,
            step=args.step_size,
            overwrite=args.overwrite,
            properties=properties,
            end=end,
            num_workers=args.num_workers,
        )

    if args.clustering:
        compare_clustering(
            results,
            WORKING_DIR / "clustering-comp.pdf",
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.agg_num:
        compare_dist(
            results,
            WORKING_DIR / "agg-num-comp.pdf",
            "Normalised aggregation numbers",
            # ylim=(0, 1.01),
            use_interval=not args.disable_interval,
            hue=AggregateProperties.VESICALITY.value,
            hue_norm=(0, 1),
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )
        compare_val(
            results,
            WORKING_DIR / "agg-num-comp-line.pdf",
            AggregateProperties.NORMALISED_AGGREGATION_NUMBERS.value,
            end=end,
            end_time=end_time,
            round_=1,
            min_cluster_size=args.min_size,
        )

    if args.cpe:
        compare_cpe(
            results,
            WORKING_DIR / "cpe-comp.pdf",
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.rog:
        compare_val(
            results,
            WORKING_DIR / "rog-comp.pdf",
            AggregateProperties.RADIUS_OF_GYRATION.value,
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.vol:
        compare_dist(
            results,
            WORKING_DIR / "vol-comp.pdf",
            AggregateProperties.VOLUME.value,
            use_interval=not args.disable_interval,
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.surf:
        compare_dist(
            results,
            WORKING_DIR / "surf-comp.pdf",
            AggregateProperties.SURFACE_AREA.value,
            use_interval=not args.disable_interval,
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )
        compare_dist(
            results,
            WORKING_DIR / "norm-surf-comp.pdf",
            AggregateProperties.SURFACE_AREA_PER_SURFACTANT.value,
            use_interval=not args.disable_interval,
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.sa_ratio:
        compare_dist(
            results,
            WORKING_DIR / "sa-ratio-comp.pdf",
            AggregateProperties.SURFACE_AREA_TO_VOLUME.value,
            use_interval=not args.disable_interval,
            ylim=(0, 1),
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.one_per_conc:
        plot_concentrations(
            results,
            properties={
                AggregateProperties.AGGREGATION_NUMBERS,
                AggregateProperties.SURFACE_AREA_PER_SURFACTANT,
                AggregateProperties.VOL_PER_SURFACTANT,
                AggregateProperties.SURFACE_AREA_TO_VOLUME,
                AggregateProperties.RADIUS_OF_GYRATION,
            },
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.total_vol:
        compare_dist(
            results,
            WORKING_DIR / "total-vol-comp.pdf",
            AggregateProperties.TOTAL_VOLUME.value,
            use_interval=not args.disable_interval,
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.soap_similarity:
        compare_soap_similarity(
            results,
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )

    if args.vesicle_contents:
        plot_vesicle_contents(
            results,
            WORKING_DIR / "vesicle-contents.pdf",
            end=end,
            end_time=end_time,
            min_cluster_size=args.min_size,
        )


if __name__ == "__main__":
    main()
