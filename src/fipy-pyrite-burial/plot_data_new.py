#!/usr/bin/env python
"""Plot diagenetic modeling data.

This module provides flexible plotting functionality for diagenetic modeling data.
The main `plot()` function can be used with default settings or with a custom
plot_description dictionary for complete control over plot structure.

It also supports overlaying measured data from a CSV file and loading plot
layouts dynamically from external Python files.

Basic Usage:
-----------
    import plot_data_new
    import pandas as pd

    df = pd.read_csv("model_output.csv")
    plot_data_new.plot(df, display_length=4, outfile="output.pdf")

Overlaying Measured Data:
------------------------
    plot_data_new.plot(df, 4, "output.pdf", measured_data_path="measured.csv")

Custom Plot Description:
-----------------------
    plot_description = {
        "first_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.c_so4, "SO4 [mmol]", {"color": "blue"}],
                [df.c_h2s, "H2S [mmol]", {"color": "red", "linestyle": "--"}],
            ],
            "left_ylabel": "Concentration [mmol/l]",
            "right": [[df.c_o2, "O2 [μmol]", {"color": "green"}]],
            "right_ylabel": "O2 [μmol/l]",
            "options-left": "set_ylim(0, 30)",  # Apply matplotlib methods
        },
    }
    plot_data_new.plot(df, 10, "output.pdf", plot_description=plot_description)
"""

import argparse
import pathlib as pl
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import importlib.util

# import matplotlib
# matplotlib.use("TkAgg")

# Maximum number of right y-axes supported per subplot
MAX_RIGHT_AXES = 10


def plot(
    df,
    display_length,
    outfile,
    show=True,
    plot_description=None,
    measured_data_path=None,
):
    """Plot data dynamically based on plot_description.

    Args
    ----
        df: DataFrame with data to plot
        display_length: Length of x-axis to display
        outfile: Output file path
        show: Whether to show the plot
        plot_description: Dictionary describing plot structure. If None, uses default structure.
            Example structure:
            {
                "first": {
                    "xaxis": [df.z, "Depth [m]"],
                    "left": [[df.c_so4, "SO4 [mmol]", {"color": "C0"}], ...],
                    "right": [[df.c_o2, "O2 [mmol]", {"color": "C2"}], ...],
                    "right2": [...],  # Additional right axes
                    "options-left": "set_ylim(0, 30)",  # Arbitrary matplotlib methods
                    "options-right": "set_ylim(0, 100)",  # For right axis
                },
                "second": {
                    "xaxis": [df.z, "Depth [m]"],
                    "left": [[df.f_so4, "f_so4"]],
                    "yscale": "log",
                    "options-left": "set_ylim(1e-10, 1e-5), set_title('Title')",
                },
            }

            The "options-left", "options-right", "options-right2", etc. keys allow
            you to specify arbitrary matplotlib method calls to apply to the corresponding
            axis. Multiple method calls can be separated by commas.
            Examples: "set_ylim(0, 100)", "set_title('My Title'), grid(True)"
        measured_data_path: Path to CSV containing measured data to overlay as scatter plots.
    """
    # Use default plot structure if none provided
    if plot_description is None:
        plot_description = _get_default_plot_description(df)

    # Filter out None subplots and count valid subplots
    valid_subplots = {k: v for k, v in plot_description.items() if v is not None}
    n_subplots = len(valid_subplots)

    if n_subplots == 0:
        raise ValueError("No valid subplots to create")

    # Load measured data if path provided
    df2 = None
    if measured_data_path:
        mpath = pl.Path(measured_data_path)
        if mpath.exists():
            df2 = pd.read_csv(mpath)
        else:
            warnings.warn(f"Measured data file not found: {measured_data_path}")

    def _get_df2_col(series_name, df2):
        if df2 is None or series_name is None:
            return None

        # Handle series_name if it's a pandas Series
        if hasattr(series_name, "name"):
            name = series_name.name
        elif isinstance(series_name, str):
            name = series_name
        else:
            return None

        # Matching strategy:
        # 1. Exact match
        # 2. Strip 'c_' prefix
        # 3. Handle isotope 'd_' vs 'c_d'
        candidates = [name]
        if name.startswith("c_"):
            candidates.append(name[2:])
        if name.startswith("d_"):
            candidates.append("c_d" + name[2:])

        for cand in candidates:
            if cand in df2.columns:
                return cand
        return None

    # Create figure and subplots
    fig, axes = plt.subplots(n_subplots, 1)
    if n_subplots == 1:
        axes = [axes]
    fig.set_size_inches(8, 2 + 2 * n_subplots)

    # Track all axes for xlim adjustment
    all_axes = []
    ax_objects = []

    # Process each subplot
    for idx, (subplot_key, subplot_config) in enumerate(valid_subplots.items()):
        ax_main = axes[idx]
        all_axes.append(ax_main)
        ax_objects.append(ax_main)

        # Get x-axis data
        xaxis_config = subplot_config.get("xaxis")
        if xaxis_config is None:
            # Default to df.z if it exists, otherwise use index
            if "z" in df.columns:
                x_data = df.z
                x_label = "Depth [m]"
            else:
                x_data = df.index
                x_label = "Index"
        else:
            x_data = xaxis_config[0]
            x_label = xaxis_config[1] if len(xaxis_config) > 1 else ""

        # Create additional y-axes as needed
        # Collect all right-side series to determine how many twin axes we need
        right_series_list = []

        # Collect series from "right", "right2", "right3", etc.
        for i in range(1, MAX_RIGHT_AXES + 1):
            key = "right" if i == 1 else f"right{i}"
            if key in subplot_config and subplot_config[key] is not None:
                series_list = subplot_config[key]
                if series_list:  # not empty
                    for series_idx, series in enumerate(series_list):
                        right_series_list.append((key, series_idx, series))

        # Create one twin axis for each right-side series
        right_axes = []
        for axis_idx, (config_key, series_idx, series) in enumerate(right_series_list):
            twin_ax = ax_main.twinx()
            all_axes.append(twin_ax)
            # Position the spine (first at 1.0, second at 1.2, third at 1.4, etc.)
            twin_ax.spines.right.set_position(("axes", 1.0 + 0.2 * axis_idx))
            right_axes.append((twin_ax, config_key, series_idx, series))

        # Plot on left axis
        left_config = subplot_config.get("left", [])
        if left_config is not None:
            left_lines = []
            left_labels = []
            for series in left_config:
                if len(series) >= 2:
                    y_data = series[0]
                    label = series[1]
                    plot_kwargs = series[2] if len(series) > 2 else {}
                    (line,) = ax_main.plot(x_data, y_data, label=label, **plot_kwargs)
                    left_lines.append(line)
                    left_labels.append(label)

                    # Plot measured data if available
                    df2_col = _get_df2_col(y_data, df2)
                    if df2_col and "z" in df2.columns:
                        ax_main.scatter(
                            df2.z,
                            df2[df2_col],
                            color=line.get_color(),
                            marker="o",
                            s=20,
                            alpha=0.6,
                            label="_nolegend_",
                        )

            # Set left axis properties
            if left_lines:
                # If there's a ylabel specified in config, use it; otherwise use first label
                ylabel = subplot_config.get(
                    "left_ylabel", left_labels[0] if left_labels else ""
                )
                ax_main.set_ylabel(ylabel)
                # Color axis by first line if only one series
                if len(left_lines) == 1:
                    ax_main.yaxis.label.set_color(left_lines[0].get_color())
                    ax_main.tick_params(axis="y", colors=left_lines[0].get_color())

        # Plot on right axes - each axis gets one series
        for twin_ax, config_key, series_idx, series in right_axes:
            if len(series) >= 2:
                y_data = series[0]
                label = series[1]
                plot_kwargs = series[2] if len(series) > 2 else {}
                (line,) = twin_ax.plot(x_data, y_data, label=label, **plot_kwargs)

                # Plot measured data if available
                df2_col = _get_df2_col(y_data, df2)
                if df2_col and "z" in df2.columns:
                    twin_ax.scatter(
                        df2.z,
                        df2[df2_col],
                        color=line.get_color(),
                        marker="o",
                        s=20,
                        alpha=0.6,
                        label="_nolegend_",
                    )

                # Set y-axis label:
                # For the first series in each config group (e.g., first in "right", first in "right2"),
                # use the explicit ylabel from config if provided (e.g., "right_ylabel").
                # For subsequent series in the same group, use the series label itself.
                # This allows users to override the first series label while automatically using
                # series labels for additional series.
                ylabel_key = f"{config_key}_ylabel"
                if series_idx == 0 and ylabel_key in subplot_config:
                    ylabel = subplot_config[ylabel_key]
                else:
                    ylabel = label
                twin_ax.set_ylabel(ylabel)

                # Color axis by the line color
                twin_ax.yaxis.label.set_color(line.get_color())
                twin_ax.tick_params(axis="y", colors=line.get_color())
                twin_ax.spines["right"].set_color(line.get_color())

        # Set x-label (usually only on bottom plot)
        if idx == n_subplots - 1:
            ax_main.set_xlabel(x_label)

        # Apply any special axis properties
        if "yscale" in subplot_config:
            ax_main.set_yscale(subplot_config["yscale"])

        # Handle legend display
        # Default to True unless explicitly set to False
        if subplot_config.get("legend", True):
            # Collect handles and labels from all axes (left and rights)
            lines = []
            labels = []

            # Get handles/labels from main axis
            l_lines, l_labels = ax_main.get_legend_handles_labels()
            lines.extend(l_lines)
            labels.extend(l_labels)

            # Get handles/labels from right axes
            for twin_ax, _, _, _ in right_axes:
                r_lines, r_labels = twin_ax.get_legend_handles_labels()
                lines.extend(r_lines)
                labels.extend(r_labels)

            # Create unified legend without frame if we have any lines
            if lines:
                # Calculate font size (80% of default)
                default_fontsize = plt.rcParams.get("legend.fontsize", 10)
                # handle if fontsize is a string 'medium', 'large' etc.
                if isinstance(default_fontsize, str):
                    # fallback if string, just use 'small' which is roughly smaller
                    fontsize = "small"
                else:
                    fontsize = default_fontsize * 0.8

                # Draw legend on the top-most axis (last twin axis) if any exist,
                # otherwise on the main axis. This prevents twin axes from drawing over the legend.
                target_ax = right_axes[-1][0] if right_axes else ax_main

                leg = target_ax.legend(
                    lines,
                    labels,
                    loc="upper right",
                    frameon=True,
                    framealpha=0.7,
                    facecolor="white",
                    edgecolor="none",
                    prop={"size": fontsize},
                )
                leg.set_zorder(100)

        # Apply arbitrary matplotlib options to left axis
        if "options-left" in subplot_config:
            _apply_matplotlib_options(ax_main, subplot_config["options-left"])

        # Apply arbitrary matplotlib options to right axes
        # Track which right axis corresponds to which config key
        right_axis_by_config = {}
        right_axis_idx = 0
        for i in range(1, MAX_RIGHT_AXES + 1):
            key = "right" if i == 1 else f"right{i}"
            if key in subplot_config and subplot_config[key] is not None:
                series_count = len(subplot_config[key]) if subplot_config[key] else 0
                if series_count > 0:
                    # Store the first axis for this config key
                    right_axis_by_config[key] = right_axis_idx
                right_axis_idx += series_count

        # Apply options to each right axis config
        for i in range(1, MAX_RIGHT_AXES + 1):
            key = "right" if i == 1 else f"right{i}"
            options_key = "options-right" if i == 1 else f"options-right{i}"

            if options_key in subplot_config and key in right_axis_by_config:
                # Get the first axis for this config
                axis_idx = right_axis_by_config[key]
                if axis_idx < len(right_axes):
                    twin_ax = right_axes[axis_idx][0]
                    _apply_matplotlib_options(twin_ax, subplot_config[options_key])

    # Adjust x-axis length for all plots
    for ax in all_axes:
        ax.set_xlim(0, display_length)

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax_objects


def load_layout_from_file(df, layout_path):
    """
    Load a plot layout from a Python file.

    Args
    ----
    df: DataFrame containing the data to plot.
    layout_path: Path to the Python file containing the layout.

    Returns
    -------
    dict: The plot description dictionary.
    """
    path = pl.Path(layout_path)
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_path}")

    # Use importlib to load the module from a file path
    spec = importlib.util.spec_from_file_location("dynamic_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_layout"):
        raise AttributeError(
            f"Layout file {layout_path} must contain a 'get_layout(df)' function."
        )

    return module.get_layout(df)


def _apply_matplotlib_options(ax, options_str):
    """Apply arbitrary matplotlib method calls to an axis.

    Args
    ----
    ax: Matplotlib axis object
    options_str: String containing matplotlib method calls, separated by commas.
                Example: "set_ylim(1e-10,1e-5), set_title('My Title')"

    The function safely parses and executes each method call on the provided axis.
    Each method call should be in the format: method_name(arg1, arg2, ...)
    Multiple calls can be separated by commas.

    Note: This function uses eval() to parse arguments. Only use with trusted input.
    """
    if not options_str or not options_str.strip():
        return

    # Split by comma to get individual method calls
    # We need to be careful with commas inside parentheses
    method_calls = []
    current_call = ""
    paren_depth = 0

    for char in options_str:
        if char == "(":
            paren_depth += 1
            current_call += char
        elif char == ")":
            paren_depth -= 1
            current_call += char
        elif char == "," and paren_depth == 0:
            # This comma is a separator between method calls
            if current_call.strip():
                method_calls.append(current_call.strip())
            current_call = ""
        else:
            current_call += char

    # Don't forget the last call
    if current_call.strip():
        method_calls.append(current_call.strip())

    # Execute each method call
    for call in method_calls:
        call = call.strip()
        if not call:
            continue

        # Parse method name and arguments
        if "(" not in call:
            # Method with no arguments
            method_name = call
            if hasattr(ax, method_name):
                getattr(ax, method_name)()
            else:
                warnings.warn(f"Axis does not have method '{method_name}', skipping")
            continue

        # Extract method name and arguments
        method_name = call[: call.index("(")]
        args_str = call[call.index("(") + 1 : call.rindex(")")]

        # Check if method exists on axis
        if not hasattr(ax, method_name):
            warnings.warn(f"Axis does not have method '{method_name}', skipping")
            continue

        # Parse arguments - evaluate the string as Python code
        # Note: This uses eval() which should only be used with trusted input
        # For user-facing applications, consider implementing a more restricted parser
        try:
            # Create a restricted evaluation context
            # Adding a trailing comma ensures the result is always a tuple,
            # even for single arguments: (x) is just x, but (x,) is a tuple
            safe_dict = {"__builtins__": {}}
            args = eval(f"({args_str},)", safe_dict)
            method = getattr(ax, method_name)
            method(*args)
        except Exception as e:
            warnings.warn(f"Failed to execute {method_name}({args_str}): {e}")


def _get_default_plot_description(df):
    """Generate default plot description for backward compatibility.

    This function creates a plot_description dictionary that replicates
    the original hard-coded plot structure from the legacy implementation.

    Args
    ----
    df: DataFrame containing the data to plot

    Returns
    -------
    dict: A plot_description dictionary with the default plot structure:
        - First subplot: Concentrations (SO4, H2S, FeS2, O2, OM, Fe)
        - Second subplot (if isotopes=True): Isotope deltas (dSO4, dH2S, dFeS2)
        - Last subplot: Reaction rates (f_o2, f_so4, f_fes2, f_h2s, f_poc)
    """
    plot_desc = {
        "first": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.c_so4, r"SO$_{4}$", {"color": "C0"}],
                [
                    df.c_h2s,
                    r"H$_{2}$S",
                    {
                        "color": "C0",
                        "linestyle": (0, (0.1, 2)),
                        "dash_capstyle": "round",
                    },
                ],
            ],
            "left_ylabel": r"SO$_{4}$ & H$_{2}$S  [mmol/l]",
            "right": [[df.c_fes2, r"FeS$_{2}$", {"color": "k"}]],
            "right_ylabel": r"FeS$_{2}$ [mmol/l]",
            "right2": [[df.c_o2, r"O$_{2}$", {"color": "C2"}]],
            "right2_ylabel": r"O$_{2}$ [$\mu$mol/l]",
            "right3": [[df.c_poc, "OM [mol/l]", {"color": "C1"}]],
            "right3_ylabel": "OM [mol/l]",
            "right4": [[df.c_fe3, "Fe [mol/l]", {"color": "C3"}]],
            "right4_ylabel": "Fe [mol/l]",
        },
    }

    # Reaction rates plot
    subplot_key = "second"
    plot_desc[subplot_key] = {
        "xaxis": [df.z, "Depth [m]"],
        "left": [
            [df.f_o2, "f_o2", {"color": "C2"}],
            [df.f_so4, "f_so4", {"color": "C0"}],
            [df.f_fes2, "f_fes2", {"color": "k"}],
            [df.f_h2s, "f_h2s", {"color": "C0", "linestyle": ":"}],
            [df.f_poc, "f_poc", {"color": "C1"}],
        ],
        "left_ylabel": r"f [mol/m$^{3}$ s$^{-1}$]",
        "yscale": "log",
        "legend": True,
    }

    return plot_desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot diagenetic modeling results with optional measured data overlay.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("input_file", help="Path to the model output CSV file.")
    parser.add_argument(
        "-d",
        "--display-length",
        dest="display_length",
        default=0,
        type=float,
        help="Depth limit for the x-axis (default: full length of data).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        default=None,
        type=str,
        help="Path for the output PDF (default: <input_file_stem>.pdf).",
    )
    parser.add_argument(
        "-m",
        "--measured-data",
        dest="measured_data",
        default=None,
        type=str,
        help="Path to CSV containing measured data to overlay as scatter points.",
    )
    parser.add_argument(
        "-l",
        "--layout",
        dest="layout_file",
        default=None,
        type=str,
        help=(
            "Path to a Python file defining the plot layout.\n"
            "The file must contain a 'get_layout(df)' function.\n\n"
            "Example layout file content:\n"
            "--------------------------------------------------\n"
            "def get_layout(df):\n"
            "    return {\n"
            "        'subplot1': {\n"
            "            'xaxis': [df.z, 'Depth [m]'],\n"
            "            'left':  [[df.c_so4, 'SO4', {'color': 'C0'}]]\n"
            "        }\n"
            "    }\n"
            "--------------------------------------------------"
        ),
    )
    parser.add_argument(
        "--hide",
        action="store_false",
        dest="show",
        help="If set, do not show the plot window, only save the PDF.",
    )
    parser.set_defaults(show=True)

    args = parser.parse_args()

    input_path = pl.Path(args.input_file)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Determine display length
    if args.display_length > 0:
        display_length = args.display_length
    elif "z" in df.columns:
        display_length = df.z.iat[-1]
    else:
        display_length = len(df)

    # Determine output file
    if args.output_file:
        outfile = pl.Path(args.output_file)
    else:
        outfile = input_path.with_suffix(".pdf")

    # Load custom layout if provided
    plt_desc = None
    if args.layout_file:
        plt_desc = load_layout_from_file(df, args.layout_file)

    plot(
        df,
        display_length,
        outfile,
        show=args.show,
        plot_description=plt_desc,
        measured_data_path=args.measured_data,
    )
    print(f"Plot generated: {outfile}")
