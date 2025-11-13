
from collections import defaultdict
import numpy as np


class KMean():
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.n_updates = 100
        self.distance = self.euclidean_distance 
    def euclidean_distance(self, a, b) -> np.float64:
        return np.sqrt(np.sum(np.square(b - a)))
    
    def calculate_centroids(self,samples,centroids):
        new_centroids_mapping = defaultdict(list)

        for sample in samples:
            distance_array = []
            for centroid in centroids:
                distance = self.distance(sample,centroid)
                distance_array.append(distance)

            sample_indices = np.argsort(distance_array)
            sample_indics = sample_indices[0]
            new_centroids_mapping[sample_indics].append(sample)

        new_centroids = [ np.mean(samples,axis=0) for _, samples in new_centroids_mapping.items()]

        if np.allclose(new_centroids,centroids):
            return new_centroids, False
        
        return new_centroids, True
    
    def apply(self,samples:np.ndarray)->np.ndarray:
        # This function take samples and return n_clusters centroid.

        # Take random n_clusters samples as initial centroids
        random_indices = np.random.choice(len(samples), self.n_clusters, replace=False)
        centroids = samples[random_indices]
        unstable_points = True
        while self.n_updates or unstable_points:
            
            centroids, unstable_points  = self.calculate_centroids(samples,centroids)
            
            self.n_updates -= 1

        return np.array(centroids)


# Example usage:
if __name__ == "__main__":
    # Generate some sample data of 1d points
    samples = np.array([1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.2, 8.0, 7.5, 9.0])

    # Generate sample data of 2d points
    samples = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])

    kmean = KMean(n_clusters=3)
    centroids = kmean.apply(samples)
    print("Centroids:", centroids)

    
