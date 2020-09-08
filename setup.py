from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dc3',
    version='0.1',
    packages=['dc3','dc3.model','dc3.eval','dc3.util','dc3.data'],
    author="Heejung Chung",
    author_email="16hchung@gmail.com",
    description="DataCentricCrystalClassification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/16hchung/DataCentricCrystalClassification",
    install_requires=['numpy',
                      'tqdm',
                      'fire',
                      'scikit-learn',
                      'scipy',
                      'pandas',
                      'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)

