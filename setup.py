import sys

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
packages = setuptools.find_namespace_packages(include=["target_prop*"])
print("PACKAGES FOUND:", packages)
print(sys.version_info)

setuptools.setup(
    name="target_prop",
    version="0.0.1",
    author="Abhinav Moudgil",
    author_email="abhinavmoudgil95@gmail.com",
    description="Distributed DTP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amoudgl/distributed-dtp",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "pytorch-lightning",
        "pytorch-lightning-bolts",
        "tqdm",
        "wandb",
        "plotly",
        "torchmetrics",
        "hydra",
    ],
)
