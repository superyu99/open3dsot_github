from setuptools import setup, find_packages

setup(
    name="vod",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
        'numpy',
        'numba',
        'matplotlib',
    ],
    python_requires='>=3.6',
)