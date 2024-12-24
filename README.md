This repository contains a Python script for performing customer segmentation on a dataset from a shopping mall.
The project utilizes unsupervised machine learning techniques to identify customer groups based on their behavior and characteristics.

- Data
The project uses the "Mall Customer Segmentation Data" from Kaggle. 

- Features

    Data Preprocessing: Cleaning, outlier removal, and standardization of the dataset.
  
    Dimensionality Reduction: Use of PCA for visualization in 2D and 3D.
  
    Clustering Algorithms: Implementation of K-Means and K-Medoids clustering.
  
    Model Evaluation: Evaluation using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
  
    Visualization: Includes elbow method plot, silhouette analysis, and 3D scatter plots for cluster visualization.
  
    New Customer Prediction: Demonstrates how to predict the cluster for a new customer.
  
- Usage
- 
    Data Analysis: The script starts by analyzing the dataset, showing basic statistics and checking for null values.
  
    Outlier Removal: Utilizes the IQR method to remove outliers.
  
    Clustering: Applies K-Means and K-Medoids clustering to find optimal customer segments.
  
    Evaluation: Assesses the clustering quality with various metrics.
  
    Visualization: Provides visual insights into the clustering results and helps in determining the optimal number of clusters.
  
    Prediction: Shows how to classify a new customer into one of the identified segments.


