from setuptools import setup, find_packages

# Read dependencies from requirements file
with open("requirements.txt", "r") as requirements_file:
    dependencies = [line.rstrip("\n") for line in requirements_file]
    dependencies = [d for d in dependencies if d[1] != "#"]

setup(
    name="resolve-scoring",
    use_scm_version={
        "tag_regex": r"^(?:[\w\-\/]+)?(?P<version>[vV]?\d+(?:\.\d+){0,2}[^\+]?)(?:\+.*)?$"
    },
    description="Resolve scoring library",
    long_description="Resolve scoring library",
    author="ICT - ISS L3 AI ",
    author_email="gva_ict_iss_l3_ai@icrc.org",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.10",
    ],
    # What does your project relate to?
    keywords="resolve,scoring",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=[]),
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # install_requires=['pyyaml','argparse'],
    install_requires=dependencies,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={},
    setup_requires=["setuptools_scm"],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and
    # allow pip to create the appropriate form of executable for the target
    # platform.
    scripts=[],
    entry_points={
        "console_scripts": [
            "resolve-scoring-cli = resolve_scoring.main:app",
        ]
    },
)
