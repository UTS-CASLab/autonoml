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
        # For data handling and representation.
        "pyarrow>=14.0",
        "numpy",
        "pandas",
        # For exporting/importing strategy files.
        "pyyaml",
        # For multiprocessing. (TODO: Revisit whether necessary.)
        "dill",
        "multiprocess",
        # For hyperparameter optimisation.
        "ConfigSpace>=0.7",
        "hpbandster",
        # For exporting/importing pipelines.
        "joblib",
        # For plotting.
        "matplotlib",
        "seaborn",
        # For machine learning components.
        "scikit-learn",
        "river",
    ],
)