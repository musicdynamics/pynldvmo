from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=7", "nbformat>=4", "nbconvert>=5", "requests>=2", "librosa", "nolds", "plotly", "pylab", "entropy", "matplotlib", "numpy>1.16", "scipy", "pywt", "sklearn", "pyentrp", "vmo"] 



setup(
    name="pynldvmo",
    version="0.0.5",
    author="Pauline Maouad",
    author_email="pmaouad@gmail.com",
    description="A package for nonlinear symbolic analysis of audio, using VMO.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/musicdynamics/pynldvmo",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
