import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))


def sample_picasso(
    data_path: str = None,
    data_type: str = "price",
    tso: str = "TNG",
    start_date: datetime = datetime(2023, 1, 1),
    end_date: datetime = datetime(2025, 1, 1),
    interval: str = "15min",
    method: str = "min",
):
    """
    Downsample 4s resolution PICASSO data to customized resolution (interval)
    :param tso: str, can use this to specify a TSO, otherwise process for all TSOs
    :param start_date:
    :param end_date:
    :param interval: downsampled resolution
    :param method: min/max/mean/median/std, by default 'min'
    :return:
    """
    data_path = Path(data_path) if data_path else Path.cwd() / "data" / "picasso"
    # data_path = os.path.join(Path.cwd(), "data", "picasso")
    dfs = []
    # start_date = datetime(2022, 5, 1)
    # end_date = datetime(2022, 5, 4)
    current_date = start_date
    type_dict = {
        "volume": "interchange",
        "price": "cbmp",
    }
    while current_date <= end_date:
        filename = (
            current_date.strftime("%Y-%m-%d") + f" picasso_{type_dict[data_type]}.csv"
        )
        file_path = os.path.join(data_path, filename)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep=";")

            # Set datetime column as index
            # df.rename(columns={"Zeit (ISO 8601)": "Time"}, inplace=True)
            df["Time"] = pd.to_datetime(
                df["Zeit (ISO 8601)"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce"
            )
            df.set_index("Time", inplace=True)
            df.drop(columns=["Zeit (ISO 8601)"], inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            if data_type == "price" and all(
                col in df.columns
                for col in [f"{tso.upper()}_POS", f"{tso.upper()}_NEG"]
            ):
                df_tso = df[[f"{tso.upper()}_POS", f"{tso.upper()}_NEG"]].copy()
            elif data_type == "volume" and f"{tso.upper()}" in df.columns:
                df_tso = df[[f"{tso.upper()}"]].copy()
            else:
                raise KeyError(
                    f"Required columns for {tso.upper()} with type='{data_type}' not found. Available: {df.columns.tolist()}"
                )

            method_func = {
                "mean": lambda x: x.mean(),
                "median": lambda x: x.median(),
                "min": lambda x: x.min(),
                "max": lambda x: x.max(),
                "std": lambda x: x.std(),
            }

            if method not in method_func:
                raise ValueError(f"Method {method} is not supported")

            # Resample the main dataframe
            df_sampled = method_func[method](df.resample(interval))

            # Resample tso columns with all methods at once
            tso_agg = df_tso.resample(interval).agg(
                ["mean", "median", "min", "max", "std"]
            )

            # Flatten column names (will create columns like 'TNG_POS_mean', 'TNG_POS_median', etc.)
            tso_agg.columns = ["_".join(col).strip() for col in tso_agg.columns.values]

            # Combine with main sampled data
            df_sampled = pd.concat([df_sampled, tso_agg], axis=1)

            dfs.append(df_sampled)
        current_date += timedelta(days=1)

    full_df = pd.concat(dfs)
    full_df = full_df.round(2)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    output_file = os.path.join(
        data_path, f"{interval}_{start_str}_{end_str}_{type_dict[data_type]}_test.csv"
    )
    full_df.to_csv(output_file)
    print(f"Successfully save the {interval} sampled PICASSO price to {output_file}.")

    picasso_prices = {}
    if tso:
        picasso_prices = {
            f"{tso}_POS": full_df[f"{tso}_POS"],
            f"{tso}_NEG": full_df[f"{tso}_NEG"],
        }

    return picasso_prices


def picasso_price():
    dfs = []
    for year in [2023, 2024]:
        data_path = f"data/picasso/15min_{year}-01-01_{year}-12-31.csv"
        df_year = pd.read_csv(os.path.join(script_dir, data_path))
        dfs.append(df_year)
    df = pd.concat(dfs, ignore_index=True)
    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df.set_index("Time", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    picasso_prices = {}
    picasso_prices["TNG_POS"] = df["TNG_POS"]
    picasso_prices["TNG_NEG"] = df["TNG_NEG"]
    return picasso_prices


def clean(col):
    return (
        col.astype(str)
        .replace("-", np.nan)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


# Actual Load
# Quarter Hour
def load(
    data_path="data/smard_TransnetBW/Actual_consumption_20220101_20250101_Quarterhour.csv",
):
    full_path = os.path.join("/Users/fxy/Documents/master_thesis/", data_path)

    df = pd.read_csv(full_path, sep=";")

    datetime_col = "Start date"
    df["Time"] = pd.to_datetime(
        df[datetime_col], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df.set_index("Time", inplace=True)
    df.index = df.index.tz_localize("Europe/Berlin", ambiguous="infer").tz_convert(
        "UTC"
    )

    grid_col = "Grid load incl. hydro pumped storage [MWh] Original resolutions"
    residual_col = "Residual load [MWh] Original resolutions"

    loads = {}
    loads["grid"] = clean(df[grid_col])
    loads["residual"] = clean(df[residual_col])

    loads["total"] = loads["grid"] + loads["residual"]

    return loads


def forecasted_load(
    data_path="data/smard_TransnetBW/Forecasted_consumption_20220101_20250101_Quarterhour.csv",
):
    full_path = os.path.join("/Users/fxy/Documents/master_thesis/", data_path)

    df = pd.read_csv(full_path, sep=";")

    datetime_col = "Start date"
    df["Time"] = pd.to_datetime(
        df[datetime_col], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df.set_index("Time", inplace=True)
    df.index = df.index.tz_localize("Europe/Berlin", ambiguous="infer").tz_convert(
        "UTC"
    )

    grid_col = "grid load [MWh] Original resolutions"
    residual_col = "Residual load [MWh] Original resolutions"

    loads = {}
    loads["grid"] = clean(df[grid_col])
    loads["residual"] = clean(df[residual_col])

    loads["total"] = loads["grid"] + loads["residual"]

    return loads


# Actual Generation
# Quarterhour
def generation(
    data_path="data/smard_TransnetBW/Actual_generation_20220101_20250101_Quarterhour.csv",
):
    full_path = os.path.join("/Users/fxy/Documents/master_thesis/", data_path)
    df = pd.read_csv(full_path, sep=";", low_memory=False)

    datetime_col = "Start date"
    df["Time"] = pd.to_datetime(
        df[datetime_col], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df.set_index("Time", inplace=True)
    # Convert local German time to UTC
    df.index = df.index.tz_localize("Europe/Berlin", ambiguous="infer").tz_convert(
        "UTC"
    )

    generation_cols = {}

    generation_cols["biomass"] = "Biomass [MWh] Original resolutions"
    generation_cols["hydropower"] = "Hydropower [MWh] Original resolutions"
    generation_cols["wind_onshore"] = "Wind onshore [MWh] Original resolutions"
    generation_cols["pv"] = "Photovoltaics [MWh] Original resolutions"
    generation_cols["other_renew"] = "Other renewable [MWh] Original resolutions"
    generation_cols["nuclear"] = "Nuclear [MWh] Original resolutions"
    generation_cols["hard_coal"] = "Hard coal [MWh] Original resolutions"
    generation_cols["fossil_gas"] = "Fossil gas [MWh] Original resolutions"
    generation_cols["hydro_pumped"] = "Hydro pumped storage [MWh] Original resolutions"
    generation_cols["other_conv"] = "Other conventional [MWh] Original resolutions"

    generations = {}
    # total = pd.Series(index=df.index, dtype=float)
    cleaned_series = []
    for i in generation_cols.keys():
        generations[i] = clean(df[generation_cols[i]])
        cleaned_series.append(generations[i])
    total = pd.concat(cleaned_series, axis=1).sum(axis=1)
    generations["total"] = total

    return generations


def forecasted_da_generation(
    data_path="data/smard_TransnetBW/Forecasted_generation_Day-Ahead_20220101_20250101_Hour_Quarterhour.csv",
):
    full_path = os.path.join("/Users/fxy/Documents/master_thesis/", data_path)
    df = pd.read_csv(full_path, sep=";", low_memory=False)

    datetime_col = "Start date"
    df["Time"] = pd.to_datetime(
        df[datetime_col], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df.set_index("Time", inplace=True)
    # Convert local German time to UTC
    df.index = df.index.tz_localize("Europe/Berlin", ambiguous="infer").tz_convert(
        "UTC"
    )

    generation_cols = {}

    generation_cols["total"] = "Total [MWh] Original resolutions"
    generation_cols["pv_and_wind"] = "Photovoltaics and wind [MWh] Original resolutions"
    generation_cols["wind_onshore"] = "Wind onshore [MWh] Original resolutions"
    generation_cols["pv"] = "Photovoltaics [MWh] Original resolutions"
    generation_cols["other"] = "Other [MWh] Original resolutions"

    generations = {}
    # total = pd.Series(index=df.index, dtype=float)
    cleaned_series = []
    for i in generation_cols.keys():
        generations[i] = clean(df[generation_cols[i]])
        cleaned_series.append(generations[i])

    return generations


def forecasted_id_generation(
    data_path="data/smard_TransnetBW/Generation_Forecast_Intraday_20220101_20250101_Quarterhour.csv",
):
    full_path = os.path.join("/Users/fxy/Documents/master_thesis/", data_path)
    df = pd.read_csv(full_path, sep=";", low_memory=False)

    datetime_col = "Start date"
    df["Time"] = pd.to_datetime(
        df[datetime_col], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df.set_index("Time", inplace=True)
    # Convert local German time to UTC
    df.index = df.index.tz_localize("Europe/Berlin", ambiguous="infer").tz_convert(
        "UTC"
    )

    generation_cols = {}

    generation_cols["pv_and_wind"] = "Photovoltaics and wind [MWh] Original resolutions"
    generation_cols["wind_onshore"] = "Wind onshore [MWh] Original resolutions"
    generation_cols["pv"] = "Photovoltaics [MWh] Original resolutions"

    generations = {}
    # total = pd.Series(index=df.index, dtype=float)
    cleaned_series = []
    for i in generation_cols.keys():
        generations[i] = clean(df[generation_cols[i]])
        cleaned_series.append(generations[i])

    return generations


# Day-ahead prices
# Hourly
def day_ahead_price(
    data_path="data/smard_TransnetBW/Day-ahead_prices_20220101_20250101_Hour.csv",
):
    full_path = os.path.join("/Users/fxy/Documents/master_thesis/", data_path)

    df = pd.read_csv(full_path, sep=";")

    datetime_col = "Start date"
    df["Time"] = pd.to_datetime(
        df[datetime_col], format="%b %d, %Y %I:%M %p", errors="coerce"
    )
    df.set_index("Time", inplace=True)
    df.index = df.index.tz_localize("Europe/Berlin", ambiguous="infer").tz_convert(
        "UTC"
    )

    price_col = "Germany/Luxembourg [€/MWh] Original resolutions"
    prices = clean(df[price_col])

    return prices


def fcr_price(data_path="/Users/fxy/Documents/master_thesis/data/Regelleistung"):
    dfs = []
    for year in [2022, 2023, 2024]:
        file_name = (
            f"RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{year}-01-01_{year}-12-31.xlsx"
        )
        full_path = os.path.join(data_path, file_name)
        df = pd.read_excel(full_path)
        df = df[df["TENDER_NUMBER"] == 1]
        dfs.append(
            df[
                [
                    "DATE_FROM",
                    "PRODUCTNAME",
                    "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]",
                ]
            ]
        )

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["DATE_FROM"] = pd.to_datetime(
        combined_df["DATE_FROM"], format="%m/%d/%y"
    )

    # PRODUCTNAME = 'NEGPOS_00_04'
    def extract_start_hour(product_str):
        try:
            parts = product_str.split("_")
            start_hour = int(parts[-2])
            return start_hour
        except:
            return None

    combined_df["START_HOUR"] = combined_df["PRODUCTNAME"].apply(extract_start_hour)
    combined_df["Time"] = combined_df["DATE_FROM"] + pd.to_timedelta(
        combined_df["START_HOUR"], unit="h"
    )
    combined_df["Time"] = combined_df["Time"].dt.tz_localize("UTC")

    combined_df.set_index("Time", inplace=True)

    df_upsampled = (
        combined_df["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]
        .resample("15min")
        .ffill()
    )  # or .interpolate() if preferred

    return df_upsampled


def afrr_capacity_price(
    data_path="/Users/fxy/Documents/master_thesis/data/Regelleistung",
    price_type: str = "average",
):
    dfs = []
    price_type_dict = {
        "average": "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]",
        "marginal": "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]",
    }
    for year in [2022, 2023, 2024]:
        file_name = (
            f"RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_{year}-01-01_{year}-12-31.xlsx"
        )
        full_path = os.path.join(data_path, file_name)

        df = pd.read_excel(full_path)

        price_column = price_type_dict[price_type]
        df = df[["DATE_FROM", "PRODUCT", price_column]]
        df[price_column] = (df[price_column] / 4).round(3)

        # Extract direction (POS or NEG)
        df["DIRECTION"] = df["PRODUCT"].str.split("_").str[0]

        # Extract block start hour (e.g., 00 from POS_00_04)
        df["START_HOUR"] = df["PRODUCT"].str.split("_").str[1].astype(int)

        # Parse date
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"], format="%m/%d/%y")

        # Create full timestamp and localize to UTC
        df["Time"] = df["DATE_FROM"] + pd.to_timedelta(df["START_HOUR"], unit="h")
        df["Time"] = df["Time"].dt.tz_localize(
            "UTC", nonexistent="NaT", ambiguous="NaT"
        )

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Pivot to get POS and NEG in separate columns
    pivot_df = combined_df.pivot(
        index="Time", columns="DIRECTION", values=price_type_dict[price_type]
    )

    # Upsample both columns to 15-minute intervals
    pivot_df = pivot_df.resample("15min").ffill()

    return pivot_df


def ip_index(
    data_path="/Users/fxy/Documents/master_thesis/data/netztransparenz/Index Ausgleichsenergiepreis [2025-05-20 17-39-54].csv",
):
    pass


# Weather
# Hourly
def weather(
    data_path="/Users/fxy/Documents/master_thesis/data/renewable_ninja/",
    years=["2022", "2023", "2024"],
):
    dfs = []
    for year in years:
        file_name = f"ninja_weather_{year}.csv"
        full_path = os.path.join(data_path, file_name)
        df = pd.read_csv(full_path, sep=",", comment="#")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    datetime_col = "time"
    combined_df["Time"] = pd.to_datetime(
        combined_df[datetime_col], format="%Y-%m-%d %H:%M", errors="coerce"
    )
    combined_df.set_index("Time", inplace=True)

    if combined_df.index.tz is None:
        combined_df.index = combined_df.index.tz_localize("UTC")

    temperature_col = "t2m"  # Celsius
    temperatures = clean(combined_df[temperature_col])

    return temperatures


def ninja_pv(
    data_path="/Users/fxy/Documents/master_thesis/data/renewable_ninja/",
    years=["2022", "2023", "2024"],
):
    dfs = []
    for year in years:
        file_name = f"ninja_pv_{year}.csv"
        full_path = os.path.join(data_path, file_name)
        df = pd.read_csv(full_path, sep=",", comment="#")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    datetime_col = "time"
    combined_df["Time"] = pd.to_datetime(
        combined_df[datetime_col], format="%Y-%m-%d %H:%M", errors="coerce"
    )
    combined_df.set_index("Time", inplace=True)

    if combined_df.index.tz is None:
        combined_df.index = combined_df.index.tz_localize("UTC")

    pv_col = " electricity in kW"  # Celsius
    pv = clean(combined_df[pv_col])

    return pv


if __name__ == "__main__":
    sample_picasso()
