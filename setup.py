from setuptools import setup, find_packages

__version__ = '0.0.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="hMDS",
    version=__version__,
    description="Python implementation of hyperbolic Multi-dimensional Scaling (hMDS)",
    package_dir={'hMDS': 'hMDS'},
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lila Boualili",
    author_email="boualili18lila@gmail.com",
    url="https://github.com/BOUALILILila/hMDS/issues",
    license="MIT",
    keywords="hmds hyperbolic embedding",
    packages=find_packages(exclude=["tests", "tests.*"]),
    zip_safe=False,
    install_requires=required,
    python_requires=">=3.6",
    project_urls={
        "Bug Tracker": "https://github.com/BOUALILILila/hMDS/issues",
        "Documentation": "https://github.com/BOUALILILila/hMDS",
        "Source Code": "https://github.com/BOUALILILila/hMDS",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)