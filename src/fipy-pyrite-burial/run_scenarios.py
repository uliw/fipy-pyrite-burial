import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from diff_lib import save_data, get_total_s_export
from run_fipy import run_model


def run_single_scenario(args):
    fe3, bt = args
    print(f"Running scenario: Fe3={fe3}, BT={bt}")
    try:
        p_dict = {"bc_fe3": fe3, "DB_depth": bt}
        (
            mp,
            c,
            k,
            species_list,
            z,
            D_mol,
            diagenetic_reactions,
            converged,
            step,
            total_time,
        ) = run_model(p_dict)

        df, fqfn = save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions)
        s, d34s = get_total_s_export(df, VCDT=mp.VCDT)
        d34s_pyrite = df.d_fes2.iloc[-1]

        return {
            "fe3": fe3,
            "bt": bt,
            "s": s,
            "d34s": d34s,
            "d34s_pyrite": d34s_pyrite,
            "converged": converged,
            "step": step,
            "total_time": total_time,
        }
    except Exception as e:
        print(f"Error in scenario Fe3={fe3}, BT={bt}: {e}")
        return {
            "fe3": fe3,
            "bt": bt,
            "s": np.nan,
            "d34s": np.nan,
            "d34s_pyrite": np.nan,
            "converged": False,
            "step": 0,
            "total_time": 0,
        }


def main():
    iron_values = np.linspace(2000, 8000, 12)
    bt_values = [0, 0.1]

    # Create parameter list
    tasks = []
    for fe3 in iron_values:
        for bt in bt_values:
            tasks.append((fe3, bt))

    # Run in parallel
    print(f"Starting parallel execution with 4 processes for {len(tasks)} scenarios...")
    with Pool(processes=4) as pool:
        results = pool.map(run_single_scenario, tasks)

    # Collect data into DataFrame
    results_df = pd.DataFrame(results)
    print("\nSimulation results summary:")
    print(results_df)

    # Save results
    results_df.to_csv("scenarios_results.csv", index=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    line_styles = ["-", "--", ":", "-."]

    for i, bt in enumerate(sorted(results_df.bt.unique())):
        df_sub = results_df[results_df.bt == bt].sort_values("fe3")
        style = line_styles[i % len(line_styles)]

        ax.plot(df_sub.fe3, df_sub.d34s, f"C0{style}", label=f"d34s (bt={bt})")
        ax.plot(
            df_sub.fe3, df_sub.d34s_pyrite, f"C1{style}", label=f"d34s_pyrite (bt={bt})"
        )

    ax.set_xlabel("Fe3 Boundary Condition [mmol/l]")
    ax.set_ylabel("delta34S [per mil]")
    ax.set_title("Sulfur Isotope Trends vs Iron Availability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scenarios_plot.pdf")
    print("\nPlot saved as scenarios_plot.pdf")
    # plt.show() # Can't show in this environment, but script is ready for it.


if __name__ == "__main__":
    main()
