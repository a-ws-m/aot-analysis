from setuptools import find_packages, setup

setup(
    name="aot",
    version="0.1",
    packages=find_packages(include="aot"),
    license="MIT",
    author="Alex Moriarty",
    author_email="amoriarty14@gmail.com",
    description="Tools for analysing Aerosol-OT.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "seaborn",
        "pandas",
        "scipy",
        "networkx",
        "matplotlib",
        "rdkit",
        "numpy",
        "pyyaml",
        "tqdm",
        "mdanalysis",
        "pytim",
        "pyvista",
    ],
    classifiers=[
        # https://pypi.org/classifiers/
    ],
    entry_points={
        "console_scripts": [
            "aot_cluster=aot.cluster:main",
            "aot_extract=scripts.extract_cluster:main",
        ],
    },
    python_requires=">=3.7",
)
