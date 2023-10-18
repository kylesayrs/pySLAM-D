from setuptools import setup, find_packages

_deps = [
    "argparse",
    "pydantic",
    "pymap3d",
    "open3d",
]

setup(
    name="Pyslamd",
    version="1.0",
    author="Kyle Sayers",
    description="",
    install_requires=_deps,
    package_dir={"": "src"},
    packages=find_packages("src", include=["pyslamd"], exclude=["*.__pycache__.*"]),
    entry_points={
        "console_scripts": [
            "pyslamd.stitch = pyslamd.main:main",
        ],
    },
)
