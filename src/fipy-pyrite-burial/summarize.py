"""
A python script to create a concise summary of the experimental steady state results.
"""
import numpy as np
import pandas as pd
import glob
import os

def get_delta(c, li, r):
    """Calculate the delta from the mass of light and heavy isotope.

    :param li: light isotope mass/concentration
    :param h: heavy isotope mass/concentration
    :param r: reference ratio

    :return : delta

    """
    #    import numpy

    with np.errstate(divide="ignore", invalid="ignore"):
        # 1. Numerical Safeguards
        # Ensure total mass 'c' is at least 'li' (light isotope) to avoid negative heavy mass
        # and prevent division by very near zero or zero.
        li_safe = np.maximum(li, 1e-30)
        c_safe = np.maximum(c, li_safe)

        h = c_safe - li_safe
        ratio = h / li_safe

        # 2. Thresholding for NaN
        # If total concentration is effectively zero, delta is undefined (NaN)
        d = np.where(c_safe < 1e-4, np.nan, 1000 * (ratio - r) / r)

        # 3. Clipping Extreme Values
        # Delta values below -999 or above extreme limits are usually numerical artifacts at trace levels.
        # -1000 is the mathematical limit for 100% light isotope (0% heavy),
        # so anything significantly below -1000 is impossible.
        d = np.clip(d, -1000.0, 1000.0)

    return d

# Export final results as
file_name = "summary.csv"

# 1 find all csv files in the local directory
csv_files = glob.glob("*.csv")
# Filter out the summary file if it already exists
csv_files = [f for f in csv_files if f != file_name]

# 2 loop over the files
results = []

# 2.1 test that the following column names are present
col_names = ["c_so4", "c_so4_32", "c_ts2", "c_ts2_32", "c_fe2_total", "c_fe3", "c_fes", "c_fes_32", "c_fes2", "c_fes2_32", "c_s0", "c_s0_32", "w", "phi"]

for current_file_name in csv_files:
    try:
        df = pd.read_csv(current_file_name)
    except Exception as e:
        print(f"Error reading {current_file_name}: {e}")
        continue

    # Check if all required columns are present
    missing_cols = [col for col in col_names if col not in df.columns]
    if missing_cols:
        print(f"Skipping {current_file_name}: Missing column '{missing_cols[0]}'")
        continue

    # 2.2 Assign values in the last row of the csv file to variables that derive their name from the column header
    last_row = df.iloc[-1]
    
    so4 = last_row["c_so4"]
    so4_32 = last_row["c_so4_32"]
    ts2 = last_row["c_ts2"]
    ts2_32 = last_row["c_ts2_32"]
    fe2_total = last_row["c_fe2_total"]
    fe3 = last_row["c_fe3"]
    fes = last_row["c_fes"]
    fes_32 = last_row["c_fes_32"]
    fes2 = last_row["c_fes2"]
    fes2_32 = last_row["c_fes2_32"]
    s0 = last_row["c_s0"]
    s0_32 = last_row["c_s0_32"]
    w = last_row["w"]
    phi = last_row["phi"]

    # 2.3 calculate the following variables in mol/m^3
    fe_total_bulk = phi * fe2_total + (1 - phi) * (fe3 + fes + fes2)

    # Yes, we need to count fes2 twice for total sulfur (FeS2)
    s_total_bulk = phi * (so4 + ts2) + (1 - phi) * (s0 + fes + 2 * fes2)

    # calculate total exit fluxes
    fe_flux = fe_total_bulk * w
    s_flux = s_total_bulk * w

    # get delta values for total flux
    s_32_total_bulk = phi * (so4_32 + ts2_32) + (1 - phi) * (s0_32 + 2 * fes2_32)
    d34S_total = get_delta(s_total_bulk, s_32_total_bulk, 0.044162589)
  
    
    # get delta values for Pyrite
    d34S_pyrite = np.nan
    if fes2_32 > 0.01:
        d34S_pyrite = get_delta(2* fes2, fes2_32, 0.044162589)
        print(f"d34S_pyrite = {d34S_pyrite}")

    # 2.4 save values as a row into results list
    results.append({
        "current_file_name": current_file_name,
        "fe_flux [mol/m^3/s]": fe_flux,
        "s_flux [mol/m^3/s]": s_flux,
        "d34S_total [mUr VCDT]": d34S_total,
        "d34S_pyrite [mUr VCDT]": d34S_pyrite
    })

# 3 save data to csv file using the file_name variable. Export without the index!
if results:
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(file_name, index=False)
    print(f"Summary saved to {file_name}")
else:
    print("No valid CSV files found to summarize.")
