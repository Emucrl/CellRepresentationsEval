[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31010/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.11+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)


![Graph](imgs/graph.png?raw=true "Cell graph on IMC roi, with CD8/PD1/PDL1 median filtering")
# SETUP
- source conda_install.sh
- source setup.sh
# RUNNING


## Define a config for the experiment you want to run (using defaults as template)
Hydra config doc can be found here for additional details https://hydra.cc/docs/intro/
- `paths`
- `graph_creation`
- `dgi`: `features_list` and overall training config
- `scripts.data_processing`: which dataset to create the graphs/train dgi/generate embeddings (correspongin paths should be set in the path files)
- `scripts.evaluations`: parameters for evaluations scripts

Mandatory fields to be updated:
- `path.dataset.cells`
- `path.clinical_info_df`
- `dgi.mlflow`
- 
## Run data processing steps to generate embeddings

- ```python scripts/embeddings_generation.py``` will run generate graphs from files in ```cfg.paths.dataset.cells```. The algorithms assume that cells are stored in csv files, with `x` and `y` columns. Graphs are then used for training a DGI encoder and then embedded, with both DGI and message passing. Datasets used for each steps can be controlled in  `conf/scripts/data_processing/defaults.yaml`. Features used as raw features for both DGI and message passing can be controlled in `cfg.dgi.training_params.features_list`.
- Hydra supports sweeper capabilities, with```hydra.sweeper.params```, to define grid search-like set of parameters to run the pipeline with (for example run the pipeline with different edge sizes). Add ```--multirun``` to trigger runs of the pipeline with all combinations of those parameters


## Evaluation scripts
Results are stored in folders defined in `conf/paths/evaluation/defaults.yaml`
parameters can be updated in `conf/scripts/evaluations/defaults.yaml`
### TLS evaluation
```python scripts/tls_evaluation.py``` 
- TLS-like masks are generated and stored in `cfg.paths.dataset.tls_masks` with one mask for each ROI that has >0 identified structures
- TLS structures are identified using DBSCAN on cells that meet the `cfg.scripts.eval.bcells_query` pandas query. We leveraged a cd20 col. One could use another column or skip this step to use precomputed binary masks.
- Using all ROIs with at least one TLS, we run a cell-level classification experiment with a linear classifier. Train/test splits are performed at ROI level

### Tumor evaluation
```python scripts/tumor_evaluation.py``` 
- In this implementation, we rely on a `tumor_mask` column in the cells csv files to generate cell-level groundtruth. One can update the `src.utils.cell_level_classification_utils.get_tumor_gt` to generate cell-level gt for each patient is a different way

### Indication evaluation
```python scripts/indication_evaluation.py``` 
- In this implementation, we rely on the file structure. Cells were stored in a `indication=XXX/datadump=YYYY/patientid_cells.csv`. We generate the ROI level groundtruth by parsing the file names.
- One can update the `src.utils.utils.find_patient_indications` to generate roi-level gt for each patient is a different way. The function should output a dataframe with patient and indication columns.

### Infiltration evaluation
```python scripts/infiltration_evaluation.py``` 
- In this implementation, we rely on a csv file with clinical information to generate ground truth label.
- One can update the `src.utils.utils.find_patient_clinical` to generate patient-level gt for each patient is a different way
- Each patient has multiple ROI/image/cell csv but a single infiltration score. We named the file with `PATIENTID_ROIX_cells.csv`. In this experiment, files with the same patient id are grouped together to generate patient level-representations. If no ROI is found in the filenames, files are treated independently as if ROI=patient. 

