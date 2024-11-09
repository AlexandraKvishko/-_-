import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# Завантаження даних
data = np.genfromtxt('A3.txt', delimiter=',')
data = data.reshape(5000, 12)  # Розділяємо на 12 каналів
N = 5000  # кількість точок
t = np.linspace(0, 10, N)  # Час запису 10 секунд

# Нормалізація даних для кращої кластеризації
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 1. Кластеризація на 11 кластерів
kmeans_11 = KMeans(n_clusters=11, random_state=42)
kmeans_11.fit(data_normalized)
labels_11 = kmeans_11.labels_
centroids_11 = kmeans_11.cluster_centers_

# Візуалізація кластерів для 11 кластерів
plt.figure(figsize=(10, 6))
plt.scatter(range(N), labels_11, c=labels_11, cmap='viridis')
plt.title('Кластеризація на 11 кластерів')
plt.xlabel('Точка даних')
plt.ylabel('Номер кластеру')

# Додаємо центри кластерів як червоні хрестики
plt.scatter(np.arange(11), centroids_11[:, 0], color='red', marker='x', label='Центри кластерів')
plt.colorbar()
plt.legend()
plt.show()

# 2. Кластеризація на 7 кластерів
kmeans_7 = KMeans(n_clusters=7, random_state=42)
kmeans_7.fit(data_normalized)
labels_7 = kmeans_7.labels_
centroids_7 = kmeans_7.cluster_centers_

# Візуалізація кластерів для 7 кластерів
plt.figure(figsize=(10, 6))
plt.scatter(range(N), labels_7, c=labels_7, cmap='viridis')
plt.title('Кластеризація на 7 кластерів')
plt.xlabel('Точка даних')
plt.ylabel('Номер кластеру')

# Додаємо центри кластерів як червоні хрестики
plt.scatter(np.arange(7), centroids_7[:, 0], color='red', marker='x', label='Центри кластерів')
plt.colorbar()
plt.legend()
plt.show()

# 3. Зниження розмірності до 3-х головних компонентів
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_normalized)

# 4. Кластеризація на 3-х компонентах (PCA)
kmeans_pca = KMeans(n_clusters=11, random_state=42)
kmeans_pca.fit(data_pca)
labels_pca = kmeans_pca.labels_
centroids_pca = kmeans_pca.cluster_centers_

# Візуалізація результатів кластеризації після PCA
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels_pca, cmap='viridis')
ax.set_title('Кластеризація після PCA на 11 кластерів')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# До
