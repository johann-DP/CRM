import argparse
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca_inertias(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Code Analytique" in df.columns:
        df = df.drop(columns=["Code Analytique"])
    num_cols = df.select_dtypes(include="number").columns
    X = StandardScaler().fit_transform(df[num_cols])
    pca = PCA().fit(X)
    eig_vals = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    cum_ratio = var_ratio.cumsum()
    return pd.DataFrame(
        {
            "Axis": range(1, len(eig_vals) + 1),
            "Eigenvalue": eig_vals,
            "%Variance": var_ratio * 100,
            "%Cumul\u00e9": cum_ratio * 100,
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export PCA inertias")
    parser.add_argument("input_file", help="CSV file to process")
    parser.add_argument(
        "--output", default="ACP_inerties.csv", help="Destination CSV file"
    )
    args = parser.parse_args(argv)

    table = compute_pca_inertias(Path(args.input_file))
    table.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
