# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import io
from google.colab import files
warnings.filterwarnings('ignore')

# Enable plotting in Colab
%matplotlib inline

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 1. DATA LOADING
print("Attempting to upload dataset. Please select 'Mall_Customers.csv' file...")
try:
    uploaded = files.upload()
    data = pd.read_csv(io.BytesIO(uploaded['Mall_Customers.csv']))
    print("File uploaded successfully!")
except:
    print("Using direct file path as fallback...")
    # Fallback if direct upload doesn't work
    data = pd.read_csv('Mall_Customers.csv')

# 2. DATA EXPLORATION
# Display basic information about the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nDescriptive Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 3. DATA PREPROCESSING AND QUALITY CHECKS
# Rename columns for clarity (if needed)
data.rename(columns={'Annual Income (k$)': 'Annual_Income', 
                      'Spending Score (1-100)': 'Spending_Score'}, inplace=True)

# Check for duplicate customer IDs
print(f"\nDuplicate CustomerIDs: {data['CustomerID'].duplicated().sum()}")

# Check for outliers using boxplots
plt.figure(figsize=(15, 5))
plt.suptitle('Boxplots to Check for Outliers', fontsize=16)

plt.subplot(1, 3, 1)
sns.boxplot(y=data['Age'])
plt.title('Age')

plt.subplot(1, 3, 2)
sns.boxplot(y=data['Annual_Income'])
plt.title('Annual Income')

plt.subplot(1, 3, 3)
sns.boxplot(y=data['Spending_Score'])
plt.title('Spending Score')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Calculate Z-scores to identify outliers
numerical_cols = ['Age', 'Annual_Income', 'Spending_Score']
z_scores = pd.DataFrame()
for col in numerical_cols:
    z_scores[f'{col}_zscore'] = (data[col] - data[col].mean()) / data[col].std()

print("\nPotential outliers (Z-score > 3):")
outliers = z_scores[(z_scores.abs() > 3).any(axis=1)]
if len(outliers) > 0:
    print(data.loc[outliers.index])
else:
    print("No extreme outliers found.")

# Decision: Use RobustScaler to handle potential outliers in a more robust way
# Create a copy of the dataset with only numerical features for clustering
X = data[numerical_cols]

# Scale the features using RobustScaler (more robust to outliers)
robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Data Summary:")
print(X_scaled_df.describe())

# 4. EXPLORATORY DATA ANALYSIS
# Distribution of individual features
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
sns.histplot(data['Age'], kde=True)
plt.title('Distribution of Age')

plt.subplot(1, 3, 2)
sns.histplot(data['Annual_Income'], kde=True)
plt.title('Distribution of Annual Income')

plt.subplot(1, 3, 3)
sns.histplot(data['Spending_Score'], kde=True)
plt.title('Distribution of Spending Score')

plt.tight_layout()
plt.show()

# Gender distribution
plt.figure(figsize=(8, 5))
gender_counts = data['Gender'].value_counts()
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')
for i, count in enumerate(gender_counts):
    plt.text(i, count + 5, f'{count} ({count/len(data)*100:.1f}%)', 
             ha='center', va='center', fontweight='bold')
plt.show()

# Feature correlation analysis
plt.figure(figsize=(10, 8))
correlation = data[numerical_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# Pairplot for all numerical features
plt.figure(figsize=(12, 10))
sns.pairplot(data, vars=numerical_cols, hue='Gender')
plt.suptitle('Pairplot of Numerical Features by Gender', y=1.02)
plt.show()

# Age vs. Spending Score with gender color coding
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Spending_Score', hue='Gender', data=data, s=100)
plt.title('Age vs. Spending Score by Gender')
plt.show()

# Annual Income vs. Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Gender', data=data, s=100)
plt.title('Annual Income vs. Spending Score by Gender')
plt.show()

# 5. FINDING OPTIMAL NUMBER OF CLUSTERS
print("\n--- Finding Optimal Number of Clusters ---")
# Metrics to evaluate
inertia = []
silhouette_scores = []
db_scores = []  # Davies-Bouldin scores
ch_scores = []  # Calinski-Harabasz scores
k_range = range(2, 11)

# Calculate metrics for different k values
for k in k_range:
    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    
    # Calculate evaluation metrics
    inertia.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    db_score = davies_bouldin_score(X_scaled, labels)
    db_scores.append(db_score)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    ch_scores.append(ch_score)
    
    print(f"K={k}, Inertia={kmeans.inertia_:.2f}, Silhouette Score={sil_score:.3f}, "
          f"Davies-Bouldin Index={db_score:.3f}, Calinski-Harabasz Index={ch_score:.2f}")

# Create DataFrame with all metrics
metrics_df = pd.DataFrame({
    'k': list(k_range),
    'Inertia': inertia,
    'Silhouette Score': silhouette_scores,
    'Davies-Bouldin Index': db_scores,
    'Calinski-Harabasz Index': ch_scores
})
print("\nMetrics Summary:")
print(metrics_df)

# Plot evaluation metrics
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis (Higher is better)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(k_range, db_scores, marker='o', linestyle='-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index (Lower is better)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(k_range, ch_scores, marker='o', linestyle='-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Calinski-Harabasz Index')
plt.title('Calinski-Harabasz Index (Higher is better)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Determine the optimal number of clusters based on metrics
# We'll use a weighted approach considering all metrics
optimal_k = metrics_df.loc[
    (metrics_df['Silhouette Score'] >= metrics_df['Silhouette Score'].max() * 0.95) &
    (metrics_df['Davies-Bouldin Index'] <= metrics_df['Davies-Bouldin Index'].min() * 1.05) &
    (metrics_df['Calinski-Harabasz Index'] >= metrics_df['Calinski-Harabasz Index'].max() * 0.95)
]['k'].min()

if np.isnan(optimal_k):
    # If no clear winner, go with best silhouette score
    optimal_k = metrics_df.loc[metrics_df['Silhouette Score'].idxmax()]['k']

print(f"\nBased on multiple metrics, the optimal number of clusters is: {optimal_k}")

# 6. IMPLEMENTING CLUSTERING ALGORITHMS
print("\n--- Implementing Multiple Clustering Algorithms ---")

# 6.1 K-Means Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
data['KMeans_Cluster'] = kmeans_labels

# 6.2 Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)
data['Hierarchical_Cluster'] = hierarchical_labels

# 6.3 DBSCAN
# Find eps parameter using nearest neighbors
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('K-Distance Graph for DBSCAN Epsilon Selection')
plt.xlabel('Data Points (sorted by distance)')
plt.ylabel('Distance to 2nd Nearest Neighbor')
plt.axhline(y=0.5, color='r', linestyle='--')  # Suggested eps value
plt.show()

# Using the elbow in the K-distance graph
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
data['DBSCAN_Cluster'] = dbscan_labels

# Count number of clusters formed by DBSCAN (excluding noise points marked as -1)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"DBSCAN found {n_clusters_dbscan} clusters and {list(dbscan_labels).count(-1)} noise points")

# 6.4 Gaussian Mixture Model
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
data['GMM_Cluster'] = gmm_labels

# 7. MODEL EVALUATION AND COMPARISON
print("\n--- Model Evaluation and Comparison ---")

# Function to evaluate clustering algorithms
def evaluate_clustering(X, labels, algorithm_name):
    if len(set(labels)) <= 1 or -1 in labels:  # Handle DBSCAN case with noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters <= 1:
            return {
                'Algorithm': algorithm_name,
                'Clusters': n_clusters,
                'Silhouette': np.nan,
                'Davies-Bouldin': np.nan,
                'Calinski-Harabasz': np.nan,
                'Noise Points': list(labels).count(-1) if -1 in labels else 0
            }
        
        # For DBSCAN, calculate metrics excluding noise points
        valid_indices = labels != -1
        if sum(valid_indices) > 1:  # Need at least 2 points for silhouette score
            sil = silhouette_score(X[valid_indices], labels[valid_indices]) if len(set(labels[valid_indices])) > 1 else np.nan
            db = davies_bouldin_score(X[valid_indices], labels[valid_indices]) if len(set(labels[valid_indices])) > 1 else np.nan
            ch = calinski_harabasz_score(X[valid_indices], labels[valid_indices]) if len(set(labels[valid_indices])) > 1 else np.nan
        else:
            sil, db, ch = np.nan, np.nan, np.nan
    else:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
    
    return {
        'Algorithm': algorithm_name,
        'Clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'Silhouette': sil,
        'Davies-Bouldin': db,
        'Calinski-Harabasz': ch,
        'Noise Points': list(labels).count(-1) if -1 in labels else 0
    }

# Evaluate all algorithms
evaluation_results = [
    evaluate_clustering(X_scaled, kmeans_labels, 'K-Means'),
    evaluate_clustering(X_scaled, hierarchical_labels, 'Hierarchical'),
    evaluate_clustering(X_scaled, dbscan_labels, 'DBSCAN'),
    evaluate_clustering(X_scaled, gmm_labels, 'GMM')
]

# Create evaluation DataFrame
eval_df = pd.DataFrame(evaluation_results)
print("Clustering Algorithm Evaluation:")
print(eval_df)

# Visualize comparison
plt.figure(figsize=(15, 10))

# Silhouette Score (higher is better)
plt.subplot(2, 2, 1)
bars = plt.bar(eval_df['Algorithm'], eval_df['Silhouette'], color='skyblue')
plt.title('Silhouette Score by Algorithm (Higher is Better)')
plt.ylim(0, max(eval_df['Silhouette'].dropna()) * 1.2)
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

# Davies-Bouldin Index (lower is better)
plt.subplot(2, 2, 2)
bars = plt.bar(eval_df['Algorithm'], eval_df['Davies-Bouldin'], color='salmon')
plt.title('Davies-Bouldin Index by Algorithm (Lower is Better)')
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

# Calinski-Harabasz Index (higher is better)
plt.subplot(2, 2, 3)
bars = plt.bar(eval_df['Algorithm'], eval_df['Calinski-Harabasz'], color='lightgreen')
plt.title('Calinski-Harabasz Index by Algorithm (Higher is Better)')
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.0f}', ha='center', va='bottom')

# Number of clusters
plt.subplot(2, 2, 4)
bars = plt.bar(eval_df['Algorithm'], eval_df['Clusters'], color='purple')
plt.title('Number of Clusters Found')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Determine the best algorithm
best_algorithm = eval_df.loc[eval_df['Silhouette'].idxmax()]['Algorithm']
print(f"\nBased on Silhouette Score, the best algorithm is: {best_algorithm}")

# Set the best labels based on evaluation
if best_algorithm == 'K-Means':
    best_labels = kmeans_labels
elif best_algorithm == 'Hierarchical':
    best_labels = hierarchical_labels
elif best_algorithm == 'DBSCAN':
    best_labels = dbscan_labels
else:  # GMM
    best_labels = gmm_labels

data['Best_Cluster'] = best_labels

# 8. VISUALIZING CLUSTERING RESULTS
print("\n--- Visualizing Clustering Results ---")

# 8.1 Generate PCA visualization for dimension reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by 2 PCA components: {sum(explained_variance)*100:.2f}%")

# 8.2 Generate t-SNE visualization for better cluster separation
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(data=X_tsne, columns=['t-SNE1', 't-SNE2'])

# Function to visualize clusters
def plot_clusters(X_2d, labels, method_name, axis_names):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=80, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.title(f'{method_name} Clustering Results')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Visualize all clustering methods with PCA
plot_clusters(X_pca, kmeans_labels, 'K-Means', [f'PC1 ({explained_variance[0]*100:.2f}%)', f'PC2 ({explained_variance[1]*100:.2f}%)'])
plot_clusters(X_pca, hierarchical_labels, 'Hierarchical', [f'PC1 ({explained_variance[0]*100:.2f}%)', f'PC2 ({explained_variance[1]*100:.2f}%)'])
plot_clusters(X_pca, dbscan_labels, 'DBSCAN', [f'PC1 ({explained_variance[0]*100:.2f}%)', f'PC2 ({explained_variance[1]*100:.2f}%)'])
plot_clusters(X_pca, gmm_labels, 'GMM', [f'PC1 ({explained_variance[0]*100:.2f}%)', f'PC2 ({explained_variance[1]*100:.2f}%)'])

# Visualize the best method with t-SNE
plot_clusters(X_tsne, best_labels, best_algorithm + ' (t-SNE)', ['t-SNE1', 't-SNE2'])

# 9. FEATURE IMPORTANCE VISUALIZATION FOR EACH CLUSTER
print("\n--- Cluster Profiling ---")

# Calculate cluster statistics
cluster_stats = data.groupby('Best_Cluster').agg({
    'Age': ['mean', 'min', 'max', 'std'],
    'Annual_Income': ['mean', 'min', 'max', 'std'],
    'Spending_Score': ['mean', 'min', 'max', 'std'],
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})

print("\nCluster Statistics (Best Algorithm):")
print(cluster_stats)

# Create parallel coordinates plot for cluster profiles
plt.figure(figsize=(14, 8))
# Create a DataFrame with the means for each feature by cluster
cluster_means = data.groupby('Best_Cluster')[numerical_cols].mean()
# Scale the means for better visualization
scaler = StandardScaler()
cluster_means_scaled = pd.DataFrame(
    scaler.fit_transform(cluster_means),
    columns=cluster_means.columns,
    index=cluster_means.index
)

# Transpose the DataFrame for parallel coordinates
cluster_means_scaled_T = cluster_means_scaled.T

# Plot parallel coordinates
from pandas.plotting import parallel_coordinates
parallel_coordinates(
    cluster_means_scaled_T.reset_index().rename(columns={'index': 'Feature'}),
    'Feature',
    colormap='viridis',
    linewidth=3
)
plt.title('Parallel Coordinates Plot of Cluster Profiles (Standardized)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Create radar chart for cluster comparison
# Prepare data for radar chart
cluster_means = data.groupby('Best_Cluster')[numerical_cols].mean()
# Normalize the means for radar chart
cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

# Create radar chart
categories = numerical_cols
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

plt.figure(figsize=(12, 12))
ax = plt.subplot(111, polar=True)

for i in sorted(data['Best_Cluster'].unique()):
    if i == -1:  # Skip noise points for DBSCAN
        continue
    values = cluster_means_normalized.loc[i].tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Cluster Characteristics Comparison', size=15)
plt.show()

# 10. CUSTOMER DISTRIBUTION VISUALIZATION
# Gender distribution by cluster
plt.figure(figsize=(14, 7))
gender_cluster = pd.crosstab(data['Best_Cluster'], data['Gender'], normalize='index')
gender_cluster.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Gender Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.legend(title='Gender')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# Function to provide marketing suggestions based on segment characteristics
def get_marketing_suggestions(age_level, income_level, spending_level):
    suggestions = []
    
    if age_level == "young":
        suggestions.append("Use social media marketing")
        suggestions.append("Promote trendy products")
    elif age_level == "middle-aged":
        suggestions.append("Focus on family-oriented marketing")
        suggestions.append("Emphasize quality and value")
    else:  # senior
        suggestions.append("Use traditional marketing channels")
        suggestions.append("Highlight comfort and reliability")
    
    if income_level == "high" and spending_level == "high":
        suggestions.append("Offer premium membership programs")
        suggestions.append("Provide exclusive shopping experiences")
    elif income_level == "high" and spending_level == "low":
        suggestions.append("Emphasize value and investment aspects")
        suggestions.append("Offer price matching guarantees")
    elif income_level == "low" and spending_level == "high":
        suggestions.append("Create loyalty rewards programs")
        suggestions.append("Offer installment payment options")
    elif income_level == "low" and spending_level == "low":
        suggestions.append("Provide budget-friendly options")
        suggestions.append("Use discount promotions")
    
    return ", ".join(suggestions)

# 11. DETAILED CLUSTER PROFILING
print("\nDetailed Cluster Profiles:")
for i in sorted(data['Best_Cluster'].unique()):
    if i == -1:  # Handle DBSCAN noise points separately
        noise_data = data[data['Best_Cluster'] == -1]
        print(f"\nNoise Points ({len(noise_data)} customers - {len(noise_data)/len(data)*100:.1f}% of total):")
        if len(noise_data) > 0:
            print(f"  Age: {noise_data['Age'].mean():.1f} years (std: {noise_data['Age'].std():.1f})")
            print(f"  Annual Income: ${noise_data['Annual_Income'].mean():.1f}k (std: ${noise_data['Annual_Income'].std():.1f}k)")
            print(f"  Spending Score: {noise_data['Spending_Score'].mean():.1f} (std: {noise_data['Spending_Score'].std():.1f})")
            print(f"  Gender distribution: {noise_data['Gender'].value_counts(normalize=True).to_dict()}")
        continue
    
    cluster_data = data[data['Best_Cluster'] == i]
    print(f"\nCluster {i} ({len(cluster_data)} customers - {len(cluster_data)/len(data)*100:.1f}% of total):")
    print(f"  Age: {cluster_data['Age'].mean():.1f} years (std: {cluster_data['Age'].std():.1f}, range: {cluster_data['Age'].min()}-{cluster_data['Age'].max()})")
    print(f"  Annual Income: ${cluster_data['Annual_Income'].mean():.1f}k (std: ${cluster_data['Annual_Income'].std():.1f}k, range: ${cluster_data['Annual_Income'].min():.1f}k-${cluster_data['Annual_Income'].max():.1f}k)")
    print(f"  Spending Score: {cluster_data['Spending_Score'].mean():.1f} (std: {cluster_data['Spending_Score'].std():.1f}, range: {cluster_data['Spending_Score'].min():.1f}-{cluster_data['Spending_Score'].max():.1f})")
    print(f"  Gender distribution: {cluster_data['Gender'].value_counts(normalize=True).to_dict()}")
    
    # Provide business interpretation based on the cluster characteristics
    interpretation = ""
    age_level = "young" if cluster_data['Age'].mean() < 30 else "middle-aged" if cluster_data['Age'].mean() < 50 else "senior"
    income_level = "low" if cluster_data['Annual_Income'].mean() < 40 else "medium" if cluster_data['Annual_Income'].mean() < 70 else "high"
    spending_level = "low" if cluster_data['Spending_Score'].mean() < 40 else "medium" if cluster_data['Spending_Score'].mean() < 70 else "high"
    
    interpretation = f"  Interpretation: This segment represents {age_level} customers with {income_level} income and {spending_level} spending habits."
    
    # Add more specific interpretations
    if income_level == "high" and spending_level == "high":
        interpretation += " These are premium customers who are willing to spend more."
    elif income_level == "high" and spending_level == "low":
        interpretation += " These customers have high income but are price-sensitive or conservative spenders."
    elif income_level == "low" and spending_level == "high":
        interpretation += " These customers prioritize shopping despite limited income."
    
    print(interpretation)
    print(f"  Marketing suggestions: {get_marketing_suggestions(age_level, income_level, spending_level)}")

# 12. SAVE RESULTS
# Save the clustered data to CSV
data.to_csv('mall_customers_clustered.csv', index=False)

print("\nClustering analysis completed. All results saved to files.")
print(f"The best clustering algorithm for this dataset is: {best_algorithm}")
