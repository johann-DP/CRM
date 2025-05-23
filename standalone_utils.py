import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from phase4v2 import (
    load_data,
    prepare_data,
    select_variables,
    sanity_check,
    handle_missing_values,
    segment_data,
)


def prepare_active_dataset(input_file: str, output_dir: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load CRM data and return the cleaned active dataset.

    Parameters
    ----------
    input_file : str
        Path to the Excel file exported from Everwin.
    output_dir : Path
        Directory where segmentation figures will be saved.

    Returns
    -------
    df_active : pd.DataFrame
        Cleaned dataframe limited to the selected variables.
    quant_vars : list of str
        Names of quantitative variables kept for the analysis.
    qual_vars : list of str
        Names of qualitative variables kept for the analysis.
    """
    df_raw = load_data(input_file)
    df_clean = prepare_data(df_raw)
    df_active, quant_vars, qual_vars = select_variables(df_clean)
    quant_vars, qual_vars = sanity_check(df_active, quant_vars, qual_vars)
    df_active = df_active[quant_vars + qual_vars]
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    # Generate simple segmentation reports
    segment_data(df_active, qual_vars, output_dir)
    return df_active, quant_vars, qual_vars
