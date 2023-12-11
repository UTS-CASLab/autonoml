from setuptools import setup, find_packages

setup(
    name="autonoml",
    author="CASLab",
    author_email="caslab@uts.edu.au",
    description="A framework for continous automated machine learning",
    packages=find_packages(),
    include_package_data=True,
    # TODO: Consider user installation options for only useful subsets of packages.
    install_requires=[
        "pyarrow>=14.0",
        "dill",
        "multiprocess",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "configspace>=0.7",
        "hpbandster",
        "scikit-learn",
        "river",
        "joblib"
    ],
)