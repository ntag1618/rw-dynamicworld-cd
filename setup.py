import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rw-dynamicworld-cd", # Replace with your own username
    version="0.0.1",
    author="Kristine Lister and Alex Kovac",
    author_email="kristine.lister@wri.org",
    description="A series of Python modules for performing change detection and post-classification processing for the Dynamic World land cover product.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wri/rw-dynamicworld-cd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

#Package requirements
# import os
# import ee
# import numpy as np
# import pandas as pd
# import random
# import json
# import calendar
# import time