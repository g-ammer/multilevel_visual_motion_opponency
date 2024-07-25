# Multilevel visual motion opponency in *Drosophila*
## Scientific publication describing a neural circuit architecture in the visual system that implements motion opponent computations at multiple network levels

![alt text](https://github.com/g-ammer/multilevel_visual_motion_opponency/blob/main/oppoency_jk.png)

## Abstract
Inhibitory interactions between opponent neuronal pathways constitute a common circuit motif across brain areas and species. However, in most cases, synaptic wiring and biophysical, cellular and network mechanisms generating opponency are unknown. Here, we combine optogenetics, voltage and calcium imaging, connectomics, electrophysiology and modeling to reveal multilevel opponent inhibition in the fly visual system. We uncover a circuit architecture in which a single cell type implements direction-selective, motion-opponent inhibition at all three network levels. This inhibition, mediated by GluClα receptors, is balanced with excitation in strength, despite tenfold fewer synapses. The different opponent network levels constitute a nested, hierarchical structure operating at increasing spatiotemporal scales. Electrophysiology and modeling suggest that distributing this computation over consecutive network levels counteracts a reduction in gain, which would result from integrating large opposing conductances at a single instance. We propose that this neural architecture provides resilience to noise while enabling high selectivity for relevant sensory information.

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
