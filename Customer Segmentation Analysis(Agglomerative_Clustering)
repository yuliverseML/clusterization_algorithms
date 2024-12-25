!pip install kaggle
!pip install optuna 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors

os.makedirs('~/.kaggle', exist_ok=True)
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python
!unzip customer-segmentation-tutorial-in-python.zip

def load_and_preprocess_data(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path)
    
    if 'CustomerID' in data.columns:
        data.drop(['CustomerID'], axis=1, inplace=True)

    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    
    data['Income_Score'] = data['Annual Income (k$)'] * data['Spending Score (1-100)'] / 100  # Пример нового признака
    
    return data

data = load_and_preprocess_data('Mall_Customers.csv')

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

def determine_optimal_clusters(data_scaled):
    silhouette_scores = []
    calinski_harabasz_scores = []
    cluster_range = range(2, 11)  # Проверяем от 2 до 10 кластеров

    for n_clusters in cluster_range:
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = cluster_model.fit_predict(data_scaled)
        
        silhouette_avg = silhouette_score(data_scaled, clusters)
        calinski_harabasz_avg = calinski_harabasz_score(data_scaled, clusters)

        silhouette_scores.append(silhouette_avg)
        calinski_harabasz_scores.append(calinski_harabasz_avg)

    return cluster_range, silhouette_scores, calinski_harabasz_scores

cluster_range, silhouette_scores, calinski_harabasz_scores = determine_optimal_clusters(data_scaled)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Средний силуэтный коэффициент')
plt.xlabel('Количество кластеров')
plt.ylabel('Силуэтный коэффициент')

plt.subplot(1, 2, 2)
plt.plot(cluster_range, calinski_harabasz_scores, marker='o')
plt.title('Индекс Калински-Харабаза')
plt.xlabel('Количество кластеров')
plt.ylabel('Индекс Калински-Харабаза')

plt.tight_layout()
plt.show()

optimal_n_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Оптимальное количество кластеров: {optimal_n_clusters}")

hierarchical_cluster = AgglomerativeClustering(n_clusters=optimal_n_clusters)
clusters = hierarchical_cluster.fit_predict(data_scaled)

data['Cluster'] = clusters
print(data.columns.tolist()) 

silhouette_avg = silhouette_score(data_scaled, clusters)
calinski_harabasz = calinski_harabasz_score(data_scaled, clusters)

print("Оценка качества кластеризации:")
print(f"Средний силуэтный коэффициент: {silhouette_avg:.4f}")
print(f"Индекс Калински-Харабаза: {calinski_harabasz:.4f}")

kmeans_model = KMeans(n_clusters=optimal_n_clusters)
kmeans_clusters = kmeans_model.fit_predict(data_scaled)

if 'Cluster' in data.columns:
    cluster_data = data[data['Cluster'] == predicted_cluster]
    print(cluster_data.describe())
else:
    print("Столбец 'Cluster' не найден.")

kmeans_silhouette_avg = silhouette_score(data_scaled, kmeans_clusters)
kmeans_calinski_harabasz = calinski_harabasz_score(data_scaled, kmeans_clusters)

print("Оценка качества KMeans:")
print(f"Средний силуэтный коэффициент: {kmeans_silhouette_avg:.4f}")
print(f"Индекс Калински-Харабаза: {kmeans_calinski_harabasz:.4f}")

def analyze_stability(data_scaled):
    kmeans_cluster_range = range(2, 11)
    stability_results = []

    for n_clusters in kmeans_cluster_range:
        kmeans_model = KMeans(n_clusters=n_clusters)
        kmeans_model.fit(data_scaled)
        
        stability_results.append(silhouette_score(data_scaled, kmeans_model.labels_))

    return kmeans_cluster_range, stability_results

kmeans_cluster_range, stability_results = analyze_stability(data_scaled)

plt.figure(figsize=(8, 5))
plt.plot(kmeans_cluster_range, stability_results, marker='o', color='orange')
plt.title('Устойчивость кластеров KMeans')
plt.xlabel('Количество кластеров')
plt.ylabel('Силуэтный коэффициент')
plt.xticks(kmeans_cluster_range)
plt.grid()
plt.show()

def predict_cluster(model, scaler, new_customer):
    new_customer['Gender'] = 1 if new_customer['Gender'] == 'Female' else 0
    
    new_customer['Income_Score'] = new_customer['Annual Income (k$)'] * new_customer['Spending Score (1-100)'] / 100
    
    new_customer_df = pd.DataFrame([new_customer])
   
    new_customer_scaled = scaler.transform(new_customer_df)

    nbrs = NearestNeighbors(n_neighbors=1).fit(data_scaled)
    _, indices = nbrs.kneighbors(new_customer_scaled)
    
    predicted_cluster = model.labels_[indices[0][0]]
    
    return predicted_cluster

new_customer = {
    'Gender': 'Female',
    'Age': 30,
    'Annual Income (k$)': 70,
    'Spending Score (1-100)': 65,
}

predicted_cluster = predict_cluster(hierarchical_cluster, scaler, new_customer)

print(f"\nНовый клиент отнесен к кластеру: {predicted_cluster}")

def visualize_new_customer(data, new_customer, predicted_cluster):
    cluster_data = data[data['Cluster'] == predicted_cluster]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], data['Age'], 
                         c=data['Cluster'], cmap='viridis', alpha=0.6)

    ax.scatter(new_customer['Annual Income (k$)'], new_customer['Spending Score (1-100)'], new_customer['Age'], 
               color='red', s=100, label='Новый клиент')

    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_zlabel('Age')
    ax.set_title('Кластеры клиентов с новым клиентом (Иерархическая кластеризация)')
    
    plt.colorbar(scatter)
    plt.legend()
    plt.show()

visualize_new_customer(data, new_customer, predicted_cluster)
