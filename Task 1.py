#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Trial project just for the basic understanding of the code and format of the code.
#Used chat gpt for creating this demo :

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and standardize the dataset
data = load_iris()
X = data.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PSO parameters
num_particles = 30
num_iterations = 100
num_clusters = 3
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive (particle) weight
c2 = 1.5  # Social (swarm) weight

# Initialize particles' positions and velocities
particles_positions = [np.random.rand(num_clusters, X_scaled.shape[1]) for _ in range(num_particles)]
particles_velocities = [np.zeros((num_clusters, X_scaled.shape[1])) for _ in range(num_particles)]
personal_best_positions = particles_positions.copy()
global_best_position = particles_positions[np.random.choice(num_particles)]

# Fitness function: Sum of squared distances from each point to its cluster centroid
def fitness(position):
    distances = np.linalg.norm(X_scaled[:, np.newaxis] - position, axis=2)
    labels = np.argmin(distances, axis=1)
    return np.sum([np.sum((X_scaled[labels == i] - position[i])**2) for i in range(num_clusters)])

# Initialize personal best fitness and global best fitness
personal_best_fitness = [fitness(p) for p in personal_best_positions]
global_best_fitness = min(personal_best_fitness)
global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]

# PSO main loop
for _ in range(num_iterations):
    for i in range(num_particles):
        # Update velocities
        r1, r2 = np.random.rand(2)
        particles_velocities[i] = (w * particles_velocities[i] +
                                   c1 * r1 * (personal_best_positions[i] - particles_positions[i]) +
                                   c2 * r2 * (global_best_position - particles_positions[i]))
        # Update positions
        particles_positions[i] += particles_velocities[i]
        # Boundary conditions
        particles_positions[i] = np.clip(particles_positions[i], 0, 1)
        # Update fitness
        current_fitness = fitness(particles_positions[i])
        if current_fitness < personal_best_fitness[i]:
            personal_best_fitness[i] = current_fitness
            personal_best_positions[i] = particles_positions[i]
        if current_fitness < global_best_fitness:
            global_best_fitness = current_fitness
            global_best_position = particles_positions[i]

# Assign labels based on the final cluster centroids
final_distances = np.linalg.norm(X_scaled[:, np.newaxis] - global_best_position, axis=2)
final_labels = np.argmin(final_distances, axis=1)

# Plot the final clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=final_labels, cmap='viridis')
plt.scatter(global_best_position[:, 0], global_best_position[:, 1], s=300, c='red')
plt.title("PSO Clustering on Iris Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# In[28]:


#Implemented Code Using a guidance from the above Chat Gpt code and enhanced the code using logic.

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create the Custom dataset
X, _ = make_blobs(n_samples=300, centers=50, cluster_std=1.0, random_state=30)

# Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PSO parameters
num_particles = 40
num_iterations = 100
num_clusters = 3
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive (particle) weight
c2 = 1.5  # Social (swarm) weight

# Initialize particles' positions and velocities
particles_positions = [np.random.rand(num_clusters, X_scaled.shape[1]) for _ in range(num_particles)]
particles_velocities = [np.zeros((num_clusters, X_scaled.shape[1])) for _ in range(num_particles)]
personal_best_positions = particles_positions.copy()
global_best_position = particles_positions[np.random.choice(num_particles)]

# Fitness function: Sum of squared distances from each point to its cluster centroid
def fitness(position):
    distances = np.linalg.norm(X_scaled[:, np.newaxis] - position, axis=2)
    labels = np.argmin(distances, axis=1)
    return np.sum([np.sum((X_scaled[labels == i] - position[i])**2) for i in range(num_clusters)])

# Initialize personal best fitness and global best fitness
personal_best_fitness = [fitness(p) for p in personal_best_positions]
global_best_fitness = min(personal_best_fitness)
global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]

# PSO main loop
for _ in range(num_iterations):
    for i in range(num_particles):
        # Update velocities
        r1, r2 = np.random.rand(2)
        particles_velocities[i] = (w * particles_velocities[i] +
                                   c1 * r1 * (personal_best_positions[i] - particles_positions[i]) +
                                   c2 * r2 * (global_best_position - particles_positions[i]))
        # Update positions
        particles_positions[i] += particles_velocities[i]
        # Boundary conditions
        particles_positions[i] = np.clip(particles_positions[i], 0, 1)
        # Update fitness
        current_fitness = fitness(particles_positions[i])
        if current_fitness < personal_best_fitness[i]:
            personal_best_fitness[i] = current_fitness
            personal_best_positions[i] = particles_positions[i]
        if current_fitness < global_best_fitness:
            global_best_fitness = current_fitness
            global_best_position = particles_positions[i]

# Assign labels based on the final cluster centroids
final_distances = np.linalg.norm(X_scaled[:, np.newaxis] - global_best_position, axis=2)
final_labels = np.argmin(final_distances, axis=1)

# Plot the final clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=final_labels, cmap='viridis')
plt.scatter(global_best_position[:, 0], global_best_position[:, 1], s=300, c='red')
plt.title("PSO Clustering on Custom Dataset")
plt.xlabel("ref 1")
plt.ylabel("ref 2")
plt.show()


# In[ ]:




