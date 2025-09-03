# Atmoz

**Atmoz** is an evolving Python package designed to streamline the handling, analysis, and visualization of atmospheric science datasets. Building upon the foundation laid by the previous [DATAS](https://github.com/moroots/DATAS) package, Atmoz aims to provide a more modular, efficient, and user-friendly toolkit for atmospheric researchers.

## Table of Contents

* [Project Overview](#project-overview)
* [Current Capabilities](#current-capabilities)
* [Directory Structure](#directory-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)

---

## Project Overview

Atmoz is developed by me (Maurice Roots). Any contributes by others will be expressly noted. The package is designed to facilitate the processing, analysis, and visualization of atmospheric science datasets from various data sources and instruments (satellite, surface, profiles, etc).

The initial release focuses on integrating and analyzing data from the **TOLNet** network, a NASA-led initiative that provides high-resolution vertically resolved ozone profiles. Future versions will expand to include support for other datasets and instruments, following the modular architecture established in this release.

## Current Capabilities

As of the latest update, Atmoz offers:

* **TOLNet Integration**: Functions for downloading, processing, and visualizing TOLNet data.
* **Data Wrangling Utilities**: Tools for cleaning and transforming raw atmospheric data into analysis-ready formats.
* **Visualization Tools**: Functions for generating plots and figures to aid in data interpretation.

The package is structured to allow easy extension, with plans to incorporate support for additional datasets and instruments in future releases.

## Directory Structure

```
Atmoz/
├── TOLNet/
│   ├── TOLNet.py           # Core TOLNet functions
│   └── TOLNet_API.ipynb    # Example notebook
├── utilities/
│   └── untar.py            # Utility for extracting tar files
├── tutorials/
│   └── data_analysis.ipynb # Example analysis notebook
├── .gitignore
├── README.md
└── pyproject.toml
```

## Installation

To install Atmoz, you can use pip:

```bash
pip install git+https://github.com/moroots/atmoz
```

## License

Atmoz is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

