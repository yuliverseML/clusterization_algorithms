!pip install kaggle
!pip install scikit-learn-extra

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python
!unzip customer-segmentation-tutorial-in-python.zip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn_extra.cluster import KMedoids  # Для K-Medoids

data = pd.read_csv('Mall_Customers.csv')

print(data.head())
print(data.describe())
print(data.isnull().sum())  # Проверка на наличие пропущенных значений

data.drop(columns=['CustomerID'], inplace=True, errors='ignore')
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

data_no_outliers = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"Количество строк после удаления выбросов: {data_no_outliers.shape[0]}")

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_no_outliers)

pca = PCA(n_components=3)  # Уменьшаем до 3-х компонент для 3D визуализации
data_pca = pca.fit_transform(data_scaled)

wcss = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = [] 

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(data_pca)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_pca, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(data_pca, kmeans.labels_))
    calinski_harabasz_scores.append(calinski_harabasz_score(data_pca, kmeans.labels_))

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', color='orange')
plt.title('Силуэтный анализ')
plt.xlabel('Количество кластеров')
plt.ylabel('Силуэтный коэффициент')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(range(2, 11), davies_bouldin_scores, marker='o', color='green')
plt.title('Индекс Дэвиса-Боулдина')
plt.xlabel('Количество кластеров')
plt.ylabel('Индекс Дэвиса-Боулдина')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(range(2, 11), calinski_harabasz_scores, marker='o', color='red')
plt.title('Индекс Калински-Харабаша')
plt.xlabel('Количество кластеров')
plt.ylabel('Индекс Калински-Харабаша')
plt.grid()

plt.tight_layout()
plt.show()

for i in range(len(silhouette_scores)):
    print(f"Количество кластеров: {i + 2}")
    print(f"Силуэтный коэффициент: {silhouette_scores[i]:.4f}")
    print(f"Индекс Дэвиса-Боулдина: {davies_bouldin_scores[i]:.4f}")
    print(f"Индекс Калински-Харабаша: {calinski_harabasz_scores[i]:.4f}")
    print("-" * 50)

optimal_clusters = np.argmax(silhouette_scores) + 2  # Оптимальное количество кластеров по силуэтному анализу

kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=500, n_init=30, random_state=42) 
clusters_kmeans = kmeans.fit_predict(data_pca)

silhouette_avg = silhouette_score(data_pca, clusters_kmeans)
davies_bouldin_avg = davies_bouldin_score(data_pca, clusters_kmeans)
calinski_harabasz_avg = calinski_harabasz_score(data_pca, clusters_kmeans)

print(f"Качество предсказания модели K-Means:")
print(f"Силуэтный коэффициент: {silhouette_avg:.4f}")
print(f"Индекс Дэвиса-Боулдина: {davies_bouldin_avg:.4f}")
print(f"Индекс Калински-Харабаша: {calinski_harabasz_avg:.4f}")

kmedoids = KMedoids(n_clusters=optimal_clusters, random_state=42)
clusters_kmedoids = kmedoids.fit_predict(data_pca)

silhouette_avg_medoids = silhouette_score(data_pca, clusters_kmedoids)
davies_bouldin_avg_medoids = davies_bouldin_score(data_pca, clusters_kmedoids)
calinski_harabasz_avg_medoids = calinski_harabasz_score(data_pca, clusters_kmedoids)

print(f"\nКачество предсказания модели K-Medoids:")
print(f"Силуэтный коэффициент: {silhouette_avg_medoids:.4f}")
print(f"Индекс Дэвиса-Боулдина: {davies_bouldin_avg_medoids:.4f}")
print(f"Индекс Калински-Харабаша: {calinski_harabasz_avg_medoids:.4f}")

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=clusters_kmeans, cmap='viridis', s=50)
ax1.set_title('Кластеры (K-Means)')
ax1.set_xlabel('PCA Component 1')
ax1.set_ylabel('PCA Component 2')
ax1.set_zlabel('PCA Component 3')

ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=clusters_kmedoids, cmap='plasma', s=50)
ax2.set_title('Кластеры (K-Medoids)')
ax2.set_xlabel('PCA Component 1')
ax2.set_ylabel('PCA Component 2')
ax2.set_zlabel('PCA Component 3')

plt.show()

new_customer = pd.DataFrame({
    'Gender': ['Female'],  
    'Age': [28],
    'Annual Income (k$)': [75],
    'Spending Score (1-100)': [80]
})

new_customer['Gender'] = new_customer['Gender'].map({'Male': 0, 'Female': 1})
new_customer_scaled = scaler.transform(new_customer)

new_customer_pca = pca.transform(new_customer_scaled)

new_customer_cluster_kmeans = kmeans.predict(new_customer_pca)
new_customer_cluster_kmedoids = kmedoids.predict(new_customer_pca)

print(f"Новый клиент принадлежит к кластеру (K-Means): {new_customer_cluster_kmeans[0]}")
print(f"Новый клиент принадлежит к кластеру (K-Medoids): {new_customer_cluster_kmedoids[0]}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=clusters_kmeans, cmap='viridis', s=50)

ax.scatter(new_customer_pca[:, 0], new_customer_pca[:, 1], new_customer_pca[:, 2], 
           c='red', s=100, marker='*', label='Новый клиент')

plt.colorbar(scatter)
ax.set_title('Кластеры клиентов с новым клиентом (K-Means)')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.legend()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=clusters_kmedoids, cmap='plasma', s=50)

ax.scatter(new_customer_pca[:, 0], new_customer_pca[:, 1], new_customer_pca[:, 2], 
           c='red', s=100, marker='*', label='Новый клиент')

plt.colorbar(scatter)
ax.set_title('Кластеры клиентов с новым клиентом (K-Medoids)')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.legend()
plt.show()


