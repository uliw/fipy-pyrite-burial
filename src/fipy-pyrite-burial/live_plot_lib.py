from __future__ import annotations

import multiprocessing as mp
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    import pathlib as pl


class LivePlotter:
    """Manages a background process for real-time plotting via a Queue."""

    def __init__(
        self,
        layout_path: str,
        display_length: float,
        measured_data_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        self.layout_path = layout_path
        self.display_length = display_length
        self.measured_data_path = measured_data_path
        self.output_path = output_path
        self._queue: mp.Queue = mp.Queue()
        self._process: Optional[mp.Process] = None

    @property
    def queue(self) -> mp.Queue:
        return self._queue

    def start(self) -> None:
        """Launch the background plotting process."""
        self._process = mp.Process(target=self._run_plot_loop, daemon=True)
        self._process.start()

    def stop(self) -> None:
        """Stop the background plotting process."""
        if self._process and self._process.is_alive():
            self._queue.put(None)  # Sentinel for exit
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()

    def _run_plot_loop(self) -> None:
        """Internal loop running in the background process."""
        import plot_data_new
        import matplotlib

        # Ensure we use an interactive backend if possible
        # Use try-except to avoid crashing if backend is already set
        try:
            # TkAgg is usually safest across platforms for interactive use
            matplotlib.use("TkAgg")
            matplotlib.rcParams["toolbar"] = "None"
        except Exception:
            pass

        plt.ion()  # Turn on interactive mode
        fig = None
        ax_objects = None
        last_df = None

        print("[LivePlotter] Background process started.")

        try:
            while True:
                # Check for new data
                try:
                    # Non-blocking get with short timeout to allow plt.pause
                    data_item = self._queue.get(timeout=0.1)
                except Exception:
                    # No data, just keep the GUI alive
                    if fig:
                        plt.pause(0.01)
                    continue

                if data_item is None:
                    print("[LivePlotter] Termination signal received.")
                    if fig and self.output_path and last_df is not None:
                        print(f"[LivePlotter] Saving final plot to {self.output_path}")
                        # Use plot_data_new.plot to ensure correct aspect ratio and sizing
                        plot_data_new.plot(
                            last_df,
                            self.display_length,
                            outfile=self.output_path,
                            show=False,
                            fig_handle=fig,
                            plot_description=plt_desc if "plt_desc" in locals() else None,
                            measured_data_path=self.measured_data_path,
                            keep_open=True,
                        )
                    break

                # data_item is expected to be a dictionary representing a DataFrame
                df = pd.DataFrame(data_item)
                last_df = df

                try:
                    plt_desc = plot_data_new.load_layout_from_file(df, self.layout_path)
                    if fig is None:
                        # First time plotting
                        fig, ax_objects = plot_data_new.plot(
                            df,
                            self.display_length,
                            outfile=None,
                            show=False,
                            plot_description=plt_desc,
                            measured_data_path=self.measured_data_path,
                            keep_open=True,
                        )
                    else:
                        # Update existing plot
                        plot_data_new.plot(
                            df,
                            self.display_length,
                            outfile=None,
                            show=False,
                            fig_handle=fig,
                            plot_description=plt_desc,
                            measured_data_path=self.measured_data_path,
                            keep_open=True,
                        )
                except Exception as e:
                    # Catch Tkinter errors specifically if possible, or any plotting error
                    if "invalid command name" in str(e):
                        print(f"[LivePlotter] GUI window closed or lost. Recreating...")
                    else:
                        print(f"[LivePlotter] Plot update error: {e}")
                    fig = None

                # Process GUI events
                if fig:
                    try:
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    except Exception as e:
                        print(f"[LivePlotter] Draw error: {e}")
                        fig = None
                    plt.pause(0.01)

        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()
            plt.close("all")
            print("[LivePlotter] Background process exiting.")


def write_to_queue_async(
    plot_queue: mp.Queue,
    mp_params: Any,
    c: Any,
    k: Any,
    species_list: list[str],
    z: Any,
    D_mol: Any,
    diagenetic_reactions: Any,
    current_dt: float,
) -> None:
    """
    Simultaneously snaps model state and sends it to the plot_queue.
    Replicates logic from diff_lib._save_data_to_disk but avoids disk I/O.
    """
    import diff_lib
    from reactions_new import equilibrium_reactions

    # 1. Capture current rates in the main thread (thread-safe snap)
    f_final, RATES = diagenetic_reactions(mp_params, c, k, diff_lib.data_container())
    f_final, RATES = equilibrium_reactions(mp_params, c, k, f_final, RATES, current_dt)

    # 2. Snapshot values (numpy arrays)
    def snap(obj):
        if hasattr(obj, "value"):
            return obj.value.copy()
        if hasattr(obj, "copy"):
            return obj.copy()
        return obj

    data = {"z": z.copy()}

    # Collect concentrations and rates
    for species_name in species_list:
        data[f"c_{species_name}"] = snap(getattr(c, species_name))
        res_tuple = getattr(f_final, species_name)
        data[f"f_{species_name}"] = snap(res_tuple[2])

    # Diffusion coefficients
    for d_name, d_val in D_mol.items():
        data[d_name] = snap(d_val)

    # Isotopes
    isotope_map = {
        "so4": "so4_32",
        "h2s": "h2s_32",
        "hs": "hs_32",
        "ts2": "ts2_32",
        "fes": "fes_32",
        "s0": "s0_32",
        "fes2": "fes2_32",
    }

    for base, iso in isotope_map.items():
        if f"c_{base}" in data and f"c_{iso}" in data:
            s_total = data[f"c_{base}"]
            if base == "fes2":
                s_total = 2.0 * s_total
            s32 = data[f"c_{iso}"]
            data[f"d_{base}"] = diff_lib.get_delta(s_total, s32, mp_params.VCDT)

    data["w"] = np.ones(len(z)) * mp_params.w
    data["phi"] = np.ones(len(z)) * mp_params.phi

    # 3. Send to queue (as a simple dict of numpy arrays, which is picklable)
    # Use put_nowait or a short timeout to avoid blocking the simulation if queue is full
    try:
        plot_queue.put_nowait(data)
    except Exception:
        # If queue is full, just skip this update for performance
        pass
