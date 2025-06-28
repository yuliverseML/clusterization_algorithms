# Mall Customer Segmentation Analysis

A comprehensive customer segmentation solution using advanced clustering techniques to identify distinct customer groups based on shopping behavior and demographics. This project analyzes the Mall Customer Segmentation Data to help retailers develop targeted marketing strategies.

The dataset used is the [**Mall Customer Segmentation Data**](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  from Kaggle.


## Models Implemented

- **K-Means Clustering**: Partitioning approach that divides customers into k distinct non-overlapping groups
- **Hierarchical Clustering**: Builds a tree of clusters using Ward's linkage method
- **DBSCAN**: Density-based algorithm that groups points in high-density regions
- **Gaussian Mixture Model (GMM)**: Probabilistic model that assumes data points are generated from a mixture of Gaussian distributions

## Features

### Data Exploration
- Comprehensive statistical analysis of demographic and behavioral attributes
- Detection and visualization of feature distributions
- Correlation analysis between customer attributes
- Gender distribution analysis
- Multi-dimensional relationship exploration through pairplots

### Data Preprocessing
- Handling of outliers using robust scaling techniques
- Z-score anomaly detection
- Feature standardization with RobustScaler for resilience against outliers
- Data quality checks for missing values and duplicates
- Detailed boxplot analysis for feature distributions

### Model Training
- Dynamic determination of optimal cluster count using multiple metrics
- Implementation of diverse clustering paradigms (centroid-based, hierarchical, density-based, probabilistic)
- Automated parameter selection (e.g., DBSCAN epsilon using k-distance graphs)
- Cross-validated model training with reproducible results

### Model Evaluation
- Multi-metric evaluation framework:
  - Silhouette Score (cluster separation quality)
  - Davies-Bouldin Index (intra-cluster density vs. separation)
  - Calinski-Harabasz Index (variance ratio criterion)
- Robust handling of special cases (noise points in DBSCAN)
- Comparative analysis across all implemented algorithms

### Visualization
- PCA and t-SNE dimensionality reduction for cluster visualization
- Interactive scatter plots of customer segments
- Parallel coordinates plots for feature importance
- Radar charts for cluster characteristic comparison
- Gender distribution analysis by cluster
- Comprehensive profiling dashboards for each segment

## Results

### Model Comparison
| Algorithm    | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Noise Points |
|--------------|----------|------------|----------------|-------------------|--------------|
| K-Means      | 7        | 0.421      | 0.787          | 131.864           | 0            |
| Hierarchical | 7        | 0.409      | 0.830          | 125.117           | 0            |
| DBSCAN       | 1        | NaN        | NaN            | NaN               | 10           |
| GMM          | 7        | 0.392      | 1.011          | 114.086           | 0            |

### Best Model
K-Means clustering demonstrated superior performance across all evaluation metrics:
- Highest Silhouette Score (0.421): Indicates well-defined, separated clusters
- Lowest Davies-Bouldin Index (0.787): Shows optimal intra-cluster density and inter-cluster separation
- Highest Calinski-Harabasz Score (131.864): Demonstrates best between-cluster to within-cluster dispersion ratio

### Feature Importance
Analysis revealed that Spending Score and Annual Income were the most influential features in determining customer segments, followed by Age. These features created natural groupings that aligned well with K-Means' spherical cluster assumption.

## Outcome

### Best Performing Model: K-Means with 7 Clusters
The optimal K-Means model identified 7 distinct customer segments:

1. **High Income, High Spenders (Premium Customers)**: Middle-aged customers with high disposable income who spend generously
2. **High Income, Low Spenders (Conservative Affluent)**: Older customers with high income but conservative spending habits
3. **Average Income, Average Spenders (Standard Customers)**: Balanced spending relative to income
4. **Low Income, High Spenders (Aspirational Shoppers)**: Younger customers who prioritize shopping despite budget constraints
5. **Low Income, Low Spenders (Budget Conscious)**: Price-sensitive customers with minimal discretionary spending
6. **Young High Spenders**: Younger demographic with above-average spending patterns
7. **Senior Conservative Shoppers**: Older customers with moderate income and conservative spending

Each segment has distinct marketing strategy implications, from premium loyalty programs to budget-friendly promotions.

## Future Work

- Implement time-series analysis to track customer segment migration over time
- Integrate purchase history data for more nuanced behavioral segmentation
- Develop a real-time segmentation API for dynamic customer classification
- Explore deep learning approaches (autoencoders) for more complex pattern recognition
- Create A/B testing framework to validate marketing strategies for each segment
- Build a recommendation system tailored to each customer segment

## Notes

- The code is optimized for Google Colab environments with interactive visualizations
- Analysis can be extended with additional demographic or behavioral data
- For large datasets, consider implementing batch processing for performance optimization
- Hyperparameter tuning can be further refined for specific business objectives

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




