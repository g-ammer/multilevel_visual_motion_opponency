# Multilevel visual motion opponency in *Drosophila*
## Scientific publication describing a neural circuit architecture in the visual system that implements motion opponent computations at multiple network levels

![alt text](https://github.com/g-ammer/multilevel_visual_motion_opponency/blob/main/oppoency_jk.png)

**Data and code for 'Multilevel visual motion opponency in *Drosophila*, Nature Neuroscience, 2023'**

This repository contains all data and code needed to reproduce the Main and Extended Data Figures of the publication and is grouped in folders that correspond to the Figures. Numerical data are generally provided in numpy-format and accompanied by analysis code provided in ipynb (Jupyter Notebook) files that allow reproduction of the manuscript's Figures.

Data analysis code is written in Python 2.7.15 and Python 3.8.8.
Note that all analysis codes necessitate the importation of the following open source Python libraries:

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.io
from scipy import stats
```

In addition, we provide two custom-written libraries in the dataset that are needed for data analysis and modelling that need to be imported in the respective notebooks:

```
import octopus as oct
import blindschleiche as bs
```

Python versions and libraries needed for executing the script are listed in the first cell of every Jupyter notebook.

Additionally, some Figures contain Excel files or Image files in png format. Supplementary Videos are provided as mp4 files.
