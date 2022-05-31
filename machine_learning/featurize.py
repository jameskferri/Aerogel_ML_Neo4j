from pathlib import Path

from tqdm import tqdm
from pandas import DataFrame, read_csv, crosstab
from numpy import isnan, array
from rdkit.Chem import MolFromSmiles, RDKFingerprint


def featurize_with_rdkit(df, compound_dict, bit_size):

    # Gather baseline fingerprint base on number of bits
    base_mol = MolFromSmiles("COC")
    base_fingerprint = RDKFingerprint(base_mol, fpSize=bit_size, maxPath=7)
    base_fingerprint = array(base_fingerprint)
    base_zeros = list([0] * len(base_fingerprint))

    for col in tqdm(list(df.columns), desc="Featurizing Dataset"):
        if df[col].dtype == "object":
            fingerprint_matrix = []
            df_col_index = df[col].index
            for compound in df[col].tolist():

                # Verify compounds is non NaN
                if isinstance(compound, str):
                    smiles = compound_dict[compound]

                    # Verify SMILES is not NaN
                    if isinstance(smiles, str):
                        mol = MolFromSmiles(smiles)
                        fingerprint = RDKFingerprint(mol, fpSize=bit_size, maxPath=7)
                        fingerprint = list(array(fingerprint))
                        fingerprint_matrix.append(fingerprint)

                    # If SMILES or compound is NaN, append base zeros
                    else:
                        fingerprint_matrix.append(base_zeros)
                else:
                    fingerprint_matrix.append(base_zeros)

            # Cast list of fingerprints to a df
            columns = []
            for i in range(bit_size):
                columns.append(f"{col}_{i}")
            fingerprint_matrix = DataFrame(fingerprint_matrix, columns=columns)
            fingerprint_matrix.index = df_col_index

            # Merge fingerprint df to main df, dropping original SMILES column
            df = df.join(fingerprint_matrix)
            df = df.drop(columns=[col])

    return df


def featurize(df, paper_id_column, bit_size=128):

    # Gather stored compound data
    raw_compounds = read_csv(Path(__file__).parent.parent / "backends/cached_compound_info.csv")
    compounds = raw_compounds['compound'].tolist()
    compounds_smiles = raw_compounds['smiles'].tolist()
    compound_dict = dict(zip(compounds, compounds_smiles))

    # Set aside grouping column data
    group_col_data = df[paper_id_column]
    df = df.drop(columns=[paper_id_column])

    # Drop Ratio Columns
    for col in df:
        if "Ratio" in col:
            df = df.drop(columns=[col])

    # Gather which str columns are SMILES columns
    def _check_is_smiles(x):
        if isinstance(x, float) and isnan(x):
            return 'maybe'
        if x in compounds:
            return 'yes'
        else:
            return 'no'
    smiles_columns = []
    for col in df:
        if df[col].dtype == "object":
            bool_series = df[col].apply(_check_is_smiles)
            bool_series = bool_series[bool_series == 'no']
            if len(bool_series) == 0:
                smiles_columns.append(col)

    # One-Hot Encode Other columns that can not be featurized with RDkit
    # https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
    for col in df:
        if df[col].dtype == "object":
            if col not in smiles_columns:
                s = df[col].explode()
                s = crosstab(s.index, s)
                new_sub_cols = []
                for sub_col in s.columns:
                    new_sub_cols.append(col + ": " + sub_col)
                s.columns = new_sub_cols
                df = df.join(s)
                df = df.drop(columns=[col])

    # Featurize compounds in each SMILES columns with a RDkit fingerprint
    df = featurize_with_rdkit(df, compound_dict, bit_size=bit_size)

    # Remove columns that have only duplicates
    drop_columns = []
    for col in df:
        if len(df[col].unique()) == 1:
            drop_columns.append(col)
    df = df.drop(columns=drop_columns)

    # Replace unknown temperature with room temperature (20 C)
    temp_columns = list(df.filter(regex="Temp").columns)
    for col in temp_columns:
        df[col] = df[col].fillna(20)

    # Replace unknown pressure with ambient pressure (0.101325 mPa)
    pressure_columns = list(df.filter(regex="Pressure").columns)
    for col in pressure_columns:
        df[col] = df[col].fillna(0.101325)

    # Replace NaN with mean of other columns
    for col in df:
        df[col] = df[col].fillna(df[col].mean())

    # Re-add grouping column data
    df = df.join(group_col_data)

    return df
