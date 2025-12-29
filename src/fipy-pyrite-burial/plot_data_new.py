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

    # Filter for valid subplot configurations (must be dictionaries)
    valid_subplots = {k: v for k, v in plot_description.items() if isinstance(v, dict)}
    n_subplots = len(valid_subplots)

    if n_subplots == 0:
        raise ValueError("No valid subplots to create")

    # Load measured data if path provided
    df2 = _load_measured_data(measured_data_path)

    # Create figure and subplots
    fig, axes = plt.subplots(n_subplots, 1)
    if n_subplots == 1:
        axes = [axes]

    # Get figure width from top-level or first subplot
    # Support both 'fig_width' and 'plot_width'
    fig_width = plot_description.get("fig_width", plot_description.get("plot_width"))
    if fig_width is None and valid_subplots:
        first_config = list(valid_subplots.values())[0]
        fig_width = first_config.get("fig_width", first_config.get("plot_width", 8))
    elif fig_width is None:
        fig_width = 8

    fig.set_size_inches(fig_width, 2 + 2 * n_subplots)

    # Track all axes for xlim adjustment
    all_axes = []
    ax_objects = []

    # Process each subplot
    for idx, (subplot_key, subplot_config) in enumerate(valid_subplots.items()):
        # Setup axes for this subplot
        ax_main, right_axes = _setup_subplot_axes(axes[idx], subplot_config)
        all_axes.append(ax_main)
        ax_objects.append(ax_main)
        for rax, _, _, _ in right_axes:
            all_axes.append(rax)

        # Get x-axis data
        x_data, x_label = _get_xaxis_data(df, subplot_config)

        # Plot on left axis
        left_config = subplot_config.get("left", [])
        if left_config:
            left_lines, left_labels = _draw_series(ax_main, x_data, left_config, df2)

            # Set left axis properties
            if left_lines:
                ylabel = subplot_config.get(
                    "left_ylabel", left_labels[0] if left_labels else ""
                )
                ax_main.set_ylabel(ylabel)
                if len(left_lines) == 1:
                    ax_main.yaxis.label.set_color(left_lines[0].get_color())
                    ax_main.tick_params(axis="y", colors=left_lines[0].get_color())

        # Plot on right axes
        for twin_ax, config_key, series_idx, series in right_axes:
            lines, labels = _draw_series(twin_ax, x_data, [series], df2)
            if not lines:
                continue

            # Only set axis properties for the first series in a group
            if series_idx == 0:
                line = lines[0]
                label = labels[0]

                ylabel_key = f"{config_key}_ylabel"
                ylabel = subplot_config.get(ylabel_key, label)
                twin_ax.set_ylabel(ylabel)
                twin_ax.yaxis.label.set_color(line.get_color())
                twin_ax.tick_params(axis="y", colors=line.get_color())
                twin_ax.spines["right"].set_color(line.get_color())

        # Set x-label (usually only on bottom plot)
        if idx == n_subplots - 1:
            ax_main.set_xlabel(x_label)

        # Apply scale and any special axis properties
        if "yscale" in subplot_config:
            ax_main.set_yscale(subplot_config["yscale"])
        if "xscale" in subplot_config:
            ax_main.set_xscale(subplot_config["xscale"])
        if "xlim" in subplot_config:
            ax_main.set_xlim(subplot_config["xlim"])
        if "ylim" in subplot_config:
            ax_main.set_ylim(subplot_config["ylim"])

        # Apply properties to right axes if specified (e.g., "right_yscale", "right2_ylim")
        seen_right_axes = set()
        for twin_ax, key, _, _ in right_axes:
            if twin_ax not in seen_right_axes:
                for prop in ["yscale", "xscale", "xlim", "ylim"]:
                    key_prop = f"{key}_{prop}"
                    if key_prop in subplot_config:
                        getattr(twin_ax, f"set_{prop}")(subplot_config[key_prop])
                seen_right_axes.add(twin_ax)

        # Handle legend display
        _add_unified_legend(ax_main, right_axes, subplot_config)

        # Apply arbitrary matplotlib options
        _apply_all_options(ax_main, right_axes, subplot_config)

    # Adjust x-axis length for all plots that don't have an explicit xlim
    for idx, (subplot_key, subplot_config) in enumerate(valid_subplots.items()):
        if "xlim" not in subplot_config:
            ax_main = ax_objects[idx]
            ax_main.set_xlim(0, display_length)

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax_objects


def _load_measured_data(measured_data_path):
    """
    Load measured data from a CSV file.

    Args
    ----
    measured_data_path: Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None: Loaded data or None if path not provided/invalid.
    """
    if not measured_data_path:
        return None

    mpath = pl.Path(measured_data_path)
    if mpath.exists():
        return pd.read_csv(mpath)

    warnings.warn(f"Measured data file not found: {measured_data_path}")
    return None


def _match_measured_column(series_name, df2):
    """
    Find a matching column in the measured data DataFrame.

    Args
    ----
    series_name: Name of the series (string or pandas Series).
    df2: DataFrame containing measured data.

    Returns
    -------
    str or None: Matched column name or None.
    """
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


def _setup_subplot_axes(ax_main, subplot_config):
    """
    Configure twin axes for a subplot based on configuration.
    Series under the same key share a twin axis.

    Args
    ----
    ax_main: The primary matplotlib axis.
    subplot_config: Configuration dictionary for this subplot.

    Returns
    -------
    tuple: (ax_main, right_axes_list) where right_axes_list is [(ax, config_key, series_idx, series), ...]
    """
    right_axes = []
    current_axis_idx = 0

    # Potential keys: "right", "right1", "right2", ...
    keys_to_check = ["right"] + [f"right{i}" for i in range(1, MAX_RIGHT_AXES + 1)]

    for key in keys_to_check:
        if key in subplot_config and subplot_config[key] is not None:
            series_list = subplot_config[key]
            if not series_list:
                continue

            # Create ONE twin axis for this key
            twin_ax = ax_main.twinx()
            # Position the spine based on the axis index
            twin_ax.spines.right.set_position(("axes", 1.0 + 0.2 * current_axis_idx))

            for series_idx, series in enumerate(series_list):
                right_axes.append((twin_ax, key, series_idx, series))

            current_axis_idx += 1

    return ax_main, right_axes


def _get_xaxis_data(df, subplot_config):
    """
    Get x-axis data and label from subplot configuration.

    Args
    ----
    df: Data source.
    subplot_config: Subplot configuration.

    Returns
    -------
    tuple: (x_data, x_label)
    """
    xaxis_config = subplot_config.get("xaxis")
    if xaxis_config is None:
        if "z" in df.columns:
            return df.z, "Depth [m]"
        return df.index, "Index"

    x_data = xaxis_config[0]
    x_label = xaxis_config[1] if len(xaxis_config) > 1 else ""
    return x_data, x_label


def _draw_series(ax, x_data, series_list, df2):
    """
    Draw lines and measured scatter points on an axis.

    Args
    ----
    ax: Matplotlib axis.
    x_data: x-axis data.
    series_list: List of series configurations.
    df2: Measured data DataFrame.

    Returns
    -------
    tuple: (lines, labels)
    """
    lines = []
    labels = []
    for series in series_list:
        if len(series) < 2:
            continue

        y_data = series[0]
        label = series[1]
        kwargs = series[2] if len(series) > 2 else {}

        (line,) = ax.plot(x_data, y_data, label=label, **kwargs)
        lines.append(line)
        labels.append(label)

        # Overlay measured data
        df2_col = _match_measured_column(y_data, df2)
        if df2_col and "z" in df2.columns:
            ax.scatter(
                df2.z,
                df2[df2_col],
                color=line.get_color(),
                marker="o",
                s=20,
                alpha=0.6,
                label="_nolegend_",
            )

    return lines, labels


def _add_unified_legend(ax_main, right_axes, subplot_config):
    """
    Collect all labels from axes and add a unified legend.

    Args
    ----
    ax_main: Primary axis.
    right_axes: List of twin axes.
    subplot_config: Subplot configuration.
    """
    if not subplot_config.get("legend", True):
        return

    raw_lines = []
    raw_labels = []

    # Main axis
    l_lines, l_labels = ax_main.get_legend_handles_labels()
    raw_lines.extend(l_lines)
    raw_labels.extend(l_labels)

    # Right axes - only collect from each unique twin axis once
    seen_axes = {ax_main}
    for twin_ax, _, _, _ in right_axes:
        if twin_ax not in seen_axes:
            r_lines, r_labels = twin_ax.get_legend_handles_labels()
            raw_lines.extend(r_lines)
            raw_labels.extend(r_labels)
            seen_axes.add(twin_ax)

    # Deduplicate by label while preserving order
    final_lines = []
    final_labels = []
    seen_labels = set()
    for line, label in zip(raw_lines, raw_labels):
        if label not in seen_labels:
            final_lines.append(line)
            final_labels.append(label)
            seen_labels.add(label)

    if final_lines:
        default_fontsize = plt.rcParams.get("legend.fontsize", 10)
        fontsize = (
            "small" if isinstance(default_fontsize, str) else default_fontsize * 0.8
        )
        target_ax = right_axes[-1][0] if right_axes else ax_main

        leg = target_ax.legend(
            final_lines,
            final_labels,
            loc="upper right",
            frameon=True,
            framealpha=0.7,
            facecolor="white",
            edgecolor="none",
            prop={"size": fontsize},
        )
        leg.set_zorder(100)


def _apply_all_options(ax_main, right_axes, subplot_config):
    """
    Apply matplotlib options to all axes in a subplot.

    Args
    ----
    ax_main: Primary axis.
    right_axes: List of twin axes.
    subplot_config: Subplot configuration.
    """
    # Left axis
    if "options-left" in subplot_config:
        _apply_matplotlib_options(ax_main, subplot_config["options-left"])

    # Right axes
    right_axis_map = {}
    current_idx = 0
    keys_to_check = ["right"] + [f"right{i}" for i in range(1, MAX_RIGHT_AXES + 1)]

    # Map each active key to the twin axis it was assigned (0, 1, 2...)
    for key in keys_to_check:
        if key in subplot_config and subplot_config[key]:
            right_axis_map[key] = current_idx
            current_idx += 1

    # Get distinct twin axes from the right_axes list (which is [(ax, key, s_idx, s), ...])
    # The axes were created in order of 'keys_to_check' in _setup_subplot_axes.
    unique_twin_axes = []
    seen_axes = set()
    for ax, key, s_idx, s in right_axes:
        if ax not in seen_axes:
            unique_twin_axes.append(ax)
            seen_axes.add(ax)

    # Apply options to the correct twin axis
    for key, axis_idx in right_axis_map.items():
        opt_key = f"options-{key}"
        if opt_key in subplot_config and axis_idx < len(unique_twin_axes):
            _apply_matplotlib_options(
                unique_twin_axes[axis_idx], subplot_config[opt_key]
            )


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
        method_name = call[: call.index("(")].strip()
        args_str = call[call.index("(") + 1 : call.rindex(")")].strip()

        # Check if method exists on axis
        if not hasattr(ax, method_name):
            warnings.warn(f"Axis does not have method '{method_name}', skipping")
            continue

        # Execute the method call using eval in a context where 'ax' is local
        try:
            # We use a restricted context for eval
            # We provide 'ax' so the evaluated string can call methods on it
            safe_dict = {"ax": ax, "__builtins__": {}}
            eval(f"ax.{method_name}({args_str})", safe_dict)
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
