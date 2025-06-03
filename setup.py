from setuptools import setup, find_packages

setup(
    name="crm-pred",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "__pycache__"]),
)
