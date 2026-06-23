# Destination Prediction Based on Traffic Knowledge Graph and GC-LSTM

This repository provides the implementation materials for the manuscript:

**Destination prediction based on traffic knowledge graph and GC-LSTM**

The code supports reproducible experiments for destination prediction using GPS trajectory data, traffic knowledge graph construction, Graph-BERT-based knowledge graph representation learning, and GC-LSTM-based spatio-temporal feature fusion.

## Project Description

Destination prediction based on partial mobile trajectories is an important task in location-based services and intelligent transportation systems. Existing methods often suffer from trajectory data sparsity and insufficient modeling of spatial-temporal dependencies. This project implements a destination prediction framework that integrates:

- GPS trajectory preprocessing;
- traffic knowledge graph construction;
- auxiliary information fusion, including weather and POI information;
- Graph-BERT-based knowledge graph representation learning;
- graph convolutional LSTM (GC-LSTM) for spatio-temporal feature fusion;
- destination prediction and evaluation using MPA@k and MHD.

The proposed method is evaluated on two public trajectory datasets: GeoLife GPS Trajectories and Porto Taxi Trajectories.


## Dataset Information

### GeoLife GPS Trajectories

The GeoLife GPS Trajectories dataset was released by Microsoft Research Asia. It contains GPS trajectories collected from users over a long period and has been widely used in trajectory mining, location prediction, and mobility behavior analysis.

Dataset source:  
https://www.microsoft.com/download/details.aspx?id=52367

Expected raw data directory:

```text
data/raw/geolife/
```

### Porto Taxi Trajectories

The Porto Taxi Trajectories dataset contains taxi trajectories collected in Porto, Portugal, and is widely used for destination prediction tasks.

Dataset source:  
https://www.kaggle.com/datasets/crailtap/taxi-trajectory/data

Expected raw data directory:

```text
data/raw/porto/
```
