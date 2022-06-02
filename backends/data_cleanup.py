from pathlib import Path

from pandas import DataFrame, read_excel, to_numeric
from numpy import nan


def df_col_range_to_mean(df: DataFrame):
    for col in df:
        if df[col].dtype == "object":
            series_range = df[col].str.split("-", expand=True)
            series_range = series_range.replace({None: nan})

            if len(series_range.columns) > 1:
                try:
                    for sub_col in series_range:
                        series_range[sub_col] = to_numeric(series_range[sub_col])
                    series_range = series_range.mean(axis=1)
                    df[col] = series_range
                except ValueError:
                    pass
    return df


def cast_columns_to_float(df: DataFrame):
    for col in df:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    return df


def cleanup(df: DataFrame, non_mut_cols: list = None):
    # Set aside columns that should not be modified
    holding_df = None
    if non_mut_cols:
        holding_df = df[non_mut_cols]
        df = df.drop(columns=non_mut_cols)

    # Delimit columns by comma
    for col in df:
        if df[col].dtype == "object":
            df_col_expanded = df[col].str.split(",", expand=True)
            df_col_expanded = DataFrame(df_col_expanded)

            if len(df_col_expanded.columns) > 1:
                new_columns = {}
                for i in range(len(df_col_expanded.columns)):
                    new_columns[i] = f"{col}_{i}"
                df_col_expanded = df_col_expanded.rename(columns=new_columns)

                df = df.join(df_col_expanded)
                df = df.drop(columns=[col])

    # Add back in columns that should not be delimited
    if holding_df is not None:
        df = df.join(holding_df)

    # Fix formatting
    df = df.replace("----", nan)
    df = df.replace(" ", " ")

    df = df.replace({None: nan})
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Cast float columns formatted as str to float
    df = cast_columns_to_float(df)

    # Turn ranges into averages of ranges
    df = df_col_range_to_mean(df)

    return df


def si_aerogel_cleanup(df: DataFrame):
    for col in df.columns:
        if "Pressure" in col:
            df[col] = df[col].replace("Ambient", 0.101325, regex=True)
        if "Temp" in col:
            df[col] = df[col].replace("Ambient", 20, regex=True)
    df = df.replace("Ice Bath", 0, regex=True)
    df = df.replace("Overnight", 24, regex=True)
    df = df.replace("<", "", regex=True)
    df = df.replace(">", "", regex=True)

    # Replace Special Characters in column names
    new_columns = []
    for col in df.columns:
        new_col = col.replace("(", "")
        new_col = new_col.replace(")", "")
        new_col = new_col.replace("#", "")
        new_col = new_col.replace("°", "")
        new_col = new_col.strip()
        new_columns.append(new_col)
    df.columns = new_columns

    # Determine weather or not Aerogel is a Xerogel or not
    def _parse(x):
        aerogels_types = ["Supercritical Drying", "Freeze Drying", "Ambient Pressure Drying"]
        if x in aerogels_types:
            return "Aerogel"
        else:
            return "Xerogel"
    df['Final Gel Type'] = df['Drying Method'].apply(_parse)

    df = df.replace({None: nan})
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Cast float columns formatted as str to float
    df = cast_columns_to_float(df)

    # Turn ranges into averages of ranges
    df = df_col_range_to_mean(df)

    return df


def fetch_si_neo4j_dataset():
    input_data = read_excel(Path(__file__).parent / "raw_si_aerogels.xlsx", sheet_name="Comprehensive")

    # Gather columns to not delimit by commas
    holding_columns = ['Authors', 'Author Emails', 'Corresponding Author', 'Corresponding Author Emails',
                       'Year', 'Cited References (#)', "Times Cited (#)", 'Final Material']
    for column in input_data.columns:
        if "notes" in column.lower() or 'title' in column.lower():
            holding_columns.append(column)

    input_data = cleanup(df=input_data, non_mut_cols=holding_columns)
    input_data = si_aerogel_cleanup(input_data)
    return input_data


def fetch_si_ml_dataset():
    input_data = fetch_si_neo4j_dataset()
    drop_columns = ['Authors', 'Author Emails', 'Corresponding Author', 'Corresponding Author Emails',
                    'Year', 'Cited References', "Times Cited"]
    for column in input_data.columns:
        if "notes" in column.lower():
            drop_columns.append(column)
    input_data = input_data.drop(columns=drop_columns)
    return input_data


if __name__ == "__main__":
    print(fetch_si_ml_dataset())
    print(fetch_si_neo4j_dataset())
