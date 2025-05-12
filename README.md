# Spatially_Clustered_Factor_Analysis (SCFA)

## Overview
This repository contains the implementation of Spatially Clustered Factor Analysis (SCFA), as proposed by the following paper.

**Jin, Y., Wakayama, T., Jiang, R. and Sugasawa, S. (2025)** [**Clustered Factor Analysis for Multivariate Spatial Data**](https://doi.org/10.1016/j.spasta.2025.100889)

## Requirements
To run this project, you need to have the following Python packages installed:
- factor-analyzer==0.4.1
- graphviz==0.20.3

You can install all required packages by running:

```bash
pip install -r requirements.txt
```


## Usage

To use SCFA in your project, follow these steps:

1. **Load your dataset**: Your dataset should contain spatial data with multiple variables.
   
2. **Initialize the SCFA model**:

   ```python
   from CFA_function import ClusteredFactorAnalysis
   from visualization import Plotter

   # Load your spatial dataset
   data = pd.read_csv('your_data.csv')
   sp = pd.read_csv('your_spatital_info.csv')

   # Initialize the SCFA model
   model = ClusteredFactorAnalysis(data, sp, factor_number = 3,)

   # Fit the model
   re=model.fit()  

   # Get the clustering results and factor loadings
   clusters = re.groups_indexs()
   loadings = model.Load_Matrix_dict()

   # Visualize the results
   Plotter_=Plotter(data,re,sp)
   Plotter_.each_group_location()

## Examples

Check out the provided Jupyter notebooks to see SCFA in action:

- `simulation.ipynb`: Shows how SCFA can be used on simulated data.
- `application.ipynb`: Demonstrates SCFA applied to real-world data.

