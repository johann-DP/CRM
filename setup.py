# setup.py
from setuptools import setup, find_packages

setup(
    name="crm-pred",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "__pycache__"]),
    install_requires=[
        "pandas>=2.2.0",
        "numpy~=1.26.1",
        "matplotlib>=3.7",
        "seaborn~=0.13.2",
        "Pillow~=11.2.1",
        "scikit-learn>=1.4.0",
        "scipy>=1.10",
        "diptest~=0.9.0",
        "matplotlib-venn",
        "prince==0.16.0",
        "openpyxl",
        "umap-learn~=0.5.7",
        "phate~=1.0.11",
        "PyYAML~=6.0.2",
        "pacmap~=0.8.0",
        "pypdf2~=3.0.1",
        "pytest~=8.3.5",
        "xgboost~=2.1.4",
        "catboost~=1.2.8",
        "statsforecast~=2.0.1",
        "prophet~=1.1.7",
        "tensorflow_cpu~=2.19.0",
        "keras~=3.10.0",
        "flake8~=7.2.0",
        "pathlib~=1.0.1",
        "joblib~=1.5.1",
        "fpdf2~=2.8.3"
    ],
)
