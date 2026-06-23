# Destination Prediction Based on Traffic Knowledge Graph and GC-LSTM

This repository provides the implementation materials for the manuscript:

**Destination prediction based on traffic knowledge graph and GC-LSTM**

The code supports reproducible experiments for destination prediction using GPS trajectory data, traffic knowledge graph construction, Graph-BERT-based knowledge graph representation learning, and GC-LSTM-based spatio-temporal feature fusion.

## 1. Project Description

Destination prediction based on partial mobile trajectories is an important task in location-based services and intelligent transportation systems. Existing methods often suffer from trajectory data sparsity and insufficient modeling of spatial-temporal dependencies. This project implements a destination prediction framework that integrates:

- GPS trajectory preprocessing;
- traffic knowledge graph construction;
- auxiliary information fusion, including weather and POI information;
- Graph-BERT-based knowledge graph representation learning;
- graph convolutional LSTM (GC-LSTM) for spatio-temporal feature fusion;
- destination prediction and evaluation using MPA@k and MHD.

The proposed method is evaluated on two public trajectory datasets: GeoLife GPS Trajectories and Porto Taxi Trajectories.

## 2. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── config/
│   ├── geolife.yaml
│   └── porto.yaml
├── data/
│   ├── raw/
│   │   ├── geolife/
│   │   └── porto/
│   ├── processed/
│   ├── triples/
│   └── external/
│       ├── poi/
│       └── weather/
├── src/
│   ├── preprocessing/
│   │   ├── clean_gps.py
│   │   ├── segment_trajectory.py
│   │   ├── extract_features.py
│   │   └── build_database.py
│   ├── knowledge_graph/
│   │   ├── extract_triples.py
│   │   ├── fuse_auxiliary_data.py
│   │   └── neo4j_import.py
│   ├── graph_bert/
│   │   ├── sampling.py
│   │   ├── embedding.py
│   │   └── train_graphbert.py
│   ├── models/
│   │   ├── gcn.py
│   │   ├── improved_lstm.py
│   │   ├── gc_lstm.py
│   │   └── destination_predictor.py
│   ├── baselines/
│   │   ├── mlp.py
│   │   ├── rnn.py
│   │   ├── at_bilstm.py
│   │   ├── tconv.py
│   │   ├── lsi_lstm.py
│   │   └── latl.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   └── utils/
│       ├── haversine.py
│       ├── io_utils.py
│       └── seed.py
├── scripts/
│   ├── run_preprocessing.sh
│   ├── run_kg_construction.sh
│   ├── run_train.sh
│   ├── run_evaluation.sh
│   └── run_ablation.sh
├── results/
│   ├── geolife/
│   └── porto/
└── LICENSE
```

## 3. Dataset Information

### 3.1 GeoLife GPS Trajectories

The GeoLife GPS Trajectories dataset was released by Microsoft Research Asia. It contains GPS trajectories collected from users over a long period and has been widely used in trajectory mining, location prediction, and mobility behavior analysis.

Dataset source:  
https://www.microsoft.com/download/details.aspx?id=52367

Expected raw data directory:

```text
data/raw/geolife/
```

### 3.2 Porto Taxi Trajectories

The Porto Taxi Trajectories dataset contains taxi trajectories collected in Porto, Portugal, and is widely used for destination prediction tasks.

Dataset source:  
https://www.kaggle.com/datasets/crailtap/taxi-trajectory/data

Expected raw data directory:

```text
data/raw/porto/
```

### 3.3 Auxiliary Data

The traffic knowledge graph also incorporates auxiliary information, including:

- weather attributes associated with trajectory dates;
- POI information associated with origin and destination regions.

Expected auxiliary data directory:

```text
data/external/
├── poi/
└── weather/
```

If auxiliary data cannot be redistributed due to licensing restrictions, please provide the download sources and preprocessing scripts.

## 4. Environment and Dependencies

The experiments were implemented using Python. The following software environment is recommended.

```text
Python >= 3.8
PyTorch >= 1.10
NumPy
pandas
scikit-learn
networkx
py2neo or neo4j
PyYAML
tqdm
matplotlib
```

If GPU acceleration is used, CUDA and the corresponding PyTorch version should be installed.

Install dependencies with:

```bash
pip install -r requirements.txt
```

An example `requirements.txt` file may include:

```text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.10.0
networkx>=2.6.0
neo4j>=5.0.0
py2neo>=2021.2.3
PyYAML>=6.0
tqdm>=4.60.0
matplotlib>=3.5.0
```

Please update the package versions according to the actual experimental environment used in the manuscript.

## 5. Methodology

The workflow consists of five main stages.

### 5.1 GPS Trajectory Preprocessing

Raw GPS records are cleaned and converted into structured trajectory data. The preprocessing procedure includes:

1. removing records with missing longitude, latitude, or timestamp values;
2. removing duplicated records;
3. sorting GPS records by trajectory identifier and timestamp;
4. calculating distance, time interval, velocity, acceleration, and deflection angle;
5. segmenting long trajectories according to a time-interval threshold;
6. clustering trajectory endpoints to generate origin and destination regions;
7. storing trajectory-point and trajectory-segment information in structured tables.

Example command:

```bash
bash scripts/run_preprocessing.sh geolife
bash scripts/run_preprocessing.sh porto
```

Output files are saved to:

```text
data/processed/
```

### 5.2 Traffic Knowledge Graph Construction

The traffic knowledge graph is constructed from structured trajectory data. The graph contains trajectory segments, trajectory points, movement attributes, origin and destination regions, weather information, and POI information.

Triples are extracted in the following forms:

```text
<tr_id/t0028, from_area, dep_area/a0010>
<tr_id/t0028, cross, tr_pointid/p000192>
<tr_id/t0028, max_temperature, 28°C>
<des_area/a0009, contains, supermarket>
```

Example command:

```bash
bash scripts/run_kg_construction.sh geolife
bash scripts/run_kg_construction.sh porto
```

Triple files are saved to:

```text
data/triples/
```

The graph can also be imported into Neo4j for visualization and query.

### 5.3 Knowledge Graph Representation Learning

Graph-BERT is used to learn vector representations of entities in the traffic knowledge graph. The model samples graph contexts using top-k intimacy sampling and incorporates node features, Weisfeiler-Lehman role embeddings, positional embeddings, and distance embeddings.

Example command:

```bash
python src/graph_bert/train_graphbert.py --config config/geolife.yaml
python src/graph_bert/train_graphbert.py --config config/porto.yaml
```

Output embeddings are saved to:

```text
results/geolife/embeddings/
results/porto/embeddings/
```

### 5.4 GC-LSTM Destination Prediction

The learned knowledge-enhanced embeddings are used as inputs to the GC-LSTM destination prediction model. Graph convolution is embedded into the recurrent gating mechanism to jointly model spatial dependencies and temporal transition patterns.

Example command:

```bash
bash scripts/run_train.sh geolife
bash scripts/run_train.sh porto
```

The main training parameters used in the manuscript are:

```text
learning rate: 0.0001
training epochs: 1000
hidden units: 16
training trajectory prefix ratio r: 0.7
```

Please check the corresponding YAML configuration files for dataset-specific settings.

### 5.5 Evaluation

The trained model is evaluated using:

- Mean Prediction Accuracy, MPA@k, with k = 5, 10, and 20;
- Mean Haversine Distance, MHD.

Example command:

```bash
bash scripts/run_evaluation.sh geolife
bash scripts/run_evaluation.sh porto
```

Evaluation results are saved to:

```text
results/geolife/
results/porto/
```

## 6. Baseline Methods

The proposed model is compared with six baseline algorithms:

1. **MLP**: uses the first five and last five GPS records of a partial trajectory as inputs to a multilayer neural network.
2. **At-BiLSTM**: uses grid latitude/longitude embedding and hierarchical quadtree embedding with an attention-based bidirectional LSTM.
3. **T-CONV**: converts partial trajectories into image-like representations and uses a convolutional neural network for destination prediction.
4. **LSI-LSTM**: uses an attention-aware LSTM model considering location semantics and location importance.
5. **LATL**: uses an adaptive attention network and LSTM to model location features and temporal dependencies.
6. **RNN**: represents urban regions using one-hot encoding and uses a recurrent neural network for destination prediction.

Example command:

```bash
python src/evaluation/evaluate.py --config config/geolife.yaml --include_baselines
python src/evaluation/evaluate.py --config config/porto.yaml --include_baselines
```

All baseline models should be evaluated using the same data split, trajectory prefix ratios, and evaluation metrics as the proposed model.

## 7. Ablation and Sensitivity Analysis

To evaluate the contribution of key components, the following model variants can be tested:

1. **Proposed w/o Traffic Knowledge Graph**: removes the traffic knowledge graph and uses only trajectory sequence features.
2. **Proposed w/o Graph-BERT**: replaces Graph-BERT embeddings with basic entity embeddings.
3. **Proposed w/o GCN**: removes graph convolution and uses the improved LSTM only.
4. **Proposed w/o auxiliary data**: removes weather and POI information from the knowledge graph.

Example command:

```bash
bash scripts/run_ablation.sh geolife
bash scripts/run_ablation.sh porto
```

Sensitivity analysis can also be conducted by varying the observed trajectory prefix ratio:

```text
r = 0.1, 0.3, 0.5, 0.7, 0.9
```

The model is trained using r = 0.7 and tested under different r values.

## 8. Reproducibility Steps

To reproduce the experiments in the manuscript, follow these steps:

### Step 1: Download datasets

Download the GeoLife and Porto datasets from the official sources listed above and place them in:

```text
data/raw/geolife/
data/raw/porto/
```

### Step 2: Prepare auxiliary data

Place POI and weather data in:

```text
data/external/poi/
data/external/weather/
```

### Step 3: Preprocess trajectory data

```bash
bash scripts/run_preprocessing.sh geolife
bash scripts/run_preprocessing.sh porto
```

### Step 4: Construct the traffic knowledge graph

```bash
bash scripts/run_kg_construction.sh geolife
bash scripts/run_kg_construction.sh porto
```

### Step 5: Train Graph-BERT embeddings

```bash
python src/graph_bert/train_graphbert.py --config config/geolife.yaml
python src/graph_bert/train_graphbert.py --config config/porto.yaml
```

### Step 6: Train the GC-LSTM destination prediction model

```bash
bash scripts/run_train.sh geolife
bash scripts/run_train.sh porto
```

### Step 7: Evaluate prediction performance

```bash
bash scripts/run_evaluation.sh geolife
bash scripts/run_evaluation.sh porto
```

### Step 8: Run baseline and ablation experiments

```bash
python src/evaluation/evaluate.py --config config/geolife.yaml --include_baselines
python src/evaluation/evaluate.py --config config/porto.yaml --include_baselines

bash scripts/run_ablation.sh geolife
bash scripts/run_ablation.sh porto
```

## 9. Expected Outputs

The main outputs include:

```text
data/processed/
    cleaned trajectory files
    trajectory segment files
    trajectory point files

data/triples/
    traffic knowledge graph triple files

results/
    trained model checkpoints
    Graph-BERT embeddings
    MPA@k results
    MHD results
    baseline comparison results
    ablation and sensitivity analysis results
```

The evaluation tables should include:

- MPA@5, MPA@10, and MPA@20 under different trajectory prefix ratios;
- MHD on GeoLife and Porto;
- baseline comparison results;
- ablation results;
- sensitivity analysis results under different r values.

## 10. Configuration

Dataset-specific parameters can be modified in:

```text
config/geolife.yaml
config/porto.yaml
```

Important parameters include:

```yaml
dataset: geolife
seed: 2024
train_ratio: 0.7
valid_ratio: 0.1
test_ratio: 0.2
learning_rate: 0.0001
epochs: 1000
hidden_units: 16
trajectory_prefix_ratio: 0.7
top_k: 20
batch_size: 64
```

Please update these parameters according to the actual experimental settings used in the final manuscript.

## 11. Notes on Data Availability

The datasets used in this study are publicly available:

- GeoLife GPS Trajectories: https://www.microsoft.com/download/details.aspx?id=52367
- Porto Taxi Trajectories: https://www.kaggle.com/datasets/crailtap/taxi-trajectory/data

Due to dataset licensing and file size limitations, raw datasets may not be redistributed in this repository. Users should download the datasets from the official sources and follow the preprocessing instructions above.

## 12. Code Availability

The source code, configuration files, dependency information, and instructions for reproducing the experiments are provided in this repository. The repository includes scripts for data preprocessing, traffic knowledge graph construction, Graph-BERT representation learning, GC-LSTM model training, baseline comparison, ablation analysis, and evaluation.

Repository URL:

```text
[Please insert GitHub, Zenodo, Figshare, or institutional repository URL here]
```

DOI, if available:

```text
[Please insert DOI here]
```

## 13. Citation

If you use this code or dataset preprocessing workflow, please cite the related manuscript:

```bibtex
@article{zhao_destination_prediction_gclstm,
  title = {Destination prediction based on traffic knowledge graph and GC-LSTM},
  author = {Zhao, Wei and Ren, Hang and Wang, Jinpeng and Liu, Haiyang and Li, Pengao},
  journal = {PeerJ Computer Science},
  year = {2025},
  note = {Manuscript under review}
}
```

Please update the citation information after publication.

## 14. License

Please specify the license for this repository, for example:

```text
MIT License
Apache License 2.0
CC BY 4.0
```

If the repository contains only code, an open-source software license such as MIT or Apache-2.0 is recommended. If the repository contains documentation or processed non-code materials, please specify the corresponding data/documentation license.

## 15. Contact

For questions about the implementation or reproducibility materials, please contact:

```text
Corresponding author: Haiyang Liu
Email: hyliu@gs.zzu.edu.cn
Institution: School of Cyber Science and Engineering, Zhengzhou University
```
