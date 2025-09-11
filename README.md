# Atmoz

**Atmoz** is an evolving Python package designed to streamline the handling, analysis, and visualization of atmospheric science datasets. Building upon the foundation laid by the previous [DATAS](https://github.com/moroots/DATAS) package, Atmoz aims to provide a more modular, efficient, and user-friendly toolkit for atmospheric researchers.

## Table of Contents

* [Project Overview](#project-overview)
* [Current Capabilities](#current-capabilities)
* [Directory Structure](#directory-structure)
* [Installation](#installation)
* [Usage](#usage)
* [License](#license)

---

## Project Overview

Atmoz is developed by me (Maurice Roots). Any contributions by others will be expressly noted. The package is designed to facilitate the processing, analysis, and visualization of atmospheric science datasets from various data sources and instruments (satellite, surface, profiles, etc).

The initial release focuses on integrating and analyzing data from the **TOLNet** network, a NASA-led initiative that provides high-resolution vertically resolved ozone profiles. Future versions will expand to include support for other datasets and instruments, following the modular architecture established in this release.

## Current Capabilities

As of the latest update, Atmoz offers:

* **TOLNet Integration**: Functions for downloading, processing, and visualizing TOLNet data.
* **Data Wrangling Utilities**: Tools for cleaning and transforming raw atmospheric data into analysis-ready formats.
* **Visualization Tools**: Functions for generating plots and figures to aid in data interpretation.

The package is structured to allow easy extension, with plans to incorporate support for additional datasets and instruments in future releases.

## Directory Structure

<!-- START FILETREE -->
 ``` 
atmoz/
├── assets/
│   ├── package_info/
│   └── watermarks/
├── data_access/
│   └── NASA.py
├── lidar/
│   └── TOLNet.py
├── models/
│   └── geos_cf.py
└── resources/
    ├── colorbars.py
    ├── debug.py
    ├── default_plot_params.py
    ├── plot_utilities.py
    └── useful_functions.py
 ``` 
<!-- END FILETREE -->

## Installation

To install Atmoz, you can use pip:

```bash
pip install git+https://github.com/moroots/atmoz
```

## License

Copyright (c) 2025 Maurice Roots

Atmoz is licensed under the [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).



