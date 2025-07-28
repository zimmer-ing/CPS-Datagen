# Generating Artificial Sensor Data for the Comparison of Unsupervised Machine Learning Methods

Official repository for the dataset generator from  
Zimmering, B.; Niggemann, O.; Hasterok, C.; Pfannstiel, E.; Ramming, D.; Pfrommer, J.:  
**Generating Artificial Sensor Data for the Comparison of Unsupervised Machine Learning Methods**,  
Sensors 2021, 21(7), 2397.  
https://doi.org/10.3390/s21072397

---

## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@Article{s21072397,
  AUTHOR = {Zimmering, Bernd and Niggemann, Oliver and Hasterok, Constanze and Pfannstiel, Erik and Ramming, Dario and Pfrommer, Julius},
  TITLE = {Generating Artificial Sensor Data for the Comparison of Unsupervised Machine Learning Methods},
  JOURNAL = {Sensors},
  VOLUME = {21},
  YEAR = {2021},
  NUMBER = {7},
  ARTICLE-NUMBER = {2397},
  URL = {https://www.mdpi.com/1424-8220/21/7/2397},
  DOI = {10.3390/s21072397}
}
```

## Installation
This code is developed and tested with Python 3.9. It is recommended to use a virtual environment for the installation. The requirements.txt file contains all necessary packages to run the code. To install the packages, run the following command:
```bash
pip install -r requirements.txt
```


## Generating Datasets
The main script for generating artificial datasets is:
```bash
python run.py
```

## Data Structure
- Generated datasets consist of multivariate time series and are intended for benchmarking unsupervised machine learning methods.
- Each dataset may contain a range of complexities (e.g., number of sensors, modes, noise level) and different types of injected anomalies (see Section 3.3 in the paper).


## Data Visualization

A Jupyter Notebook is provided in the Notebooks folder for basic exploration and visualization:

### Notebooks/Visualize_Datagen.ipynb
- This notebook allows you to load and plot generated datasets for inspection and quick analysis.
- Note: Some packages required for the notebook (e.g., matplotlib, jupyter, seaborn) are not included in requirements.txt and may need to be installed manually.

To use the notebook, run:
```bash
jupyter notebook Notebooks/Visualize_Datagen.ipynb
```

## Creating New Datasets
To create new datasets, you can use the `gen_parameters.py` file.

## License
This code is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions, feedback, or collaboration requests, please contact  
**Bernd Zimmering** – bernd.zimmering@hsu-hh.de