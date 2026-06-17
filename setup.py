from setuptools import setup, find_packages

setup(
    name="diffgenslm",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24",
        "h5py>=3.8",
        "sentencepiece>=0.1.99",
        "biopython>=1.81",
        "tokenizers>=0.15",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "wandb>=0.15",
    ],
    extras_require={
        "mpi": ["mpi4py>=3.1"],
        "dev": ["pytest>=7.0", "black", "isort"],
    },
)
