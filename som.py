import numpy as np

class SelfOrganizingMap:
    """
    Self-Organizing Map (SOM) class to train and cluster data based on similarity.

    Attributes:
    - width: int
        Width of the SOM grid.
    - height: int
        Height of the SOM grid.
    - max_iterations: int
        Maximum number of iterations for training.
    - learning_rate: float
        Initial learning rate for training, decays over time.
    - neighborhood_radius: float
        Initial neighborhood radius for BMU influence, decays over time.
    - weights: np.ndarray
        The weight matrix of the SOM, initialized randomly.
    """

    def __init__(self,  width, height, max_iterations, learning_rate=0.1):
        """
        Initializes the SOM with grid dimensions, iterations, and learning rate.

        Parameters:
        - width: int
            Width of the SOM grid.
        - height: int
            Height of the SOM grid.
        - max_iterations: int
            Maximum number of training iterations.
        - learning_rate: float, optional
            Initial learning rate (default is 0.1).
        """
        self.width = width
        self.height = height
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.neighborhood_radius = max(width, height) / 2  # Initial radius based on grid size
        self.weights = np.random.random((width, height, 3))  # Initialize weights randomly

    def train(self, input_data):
        """
        Train the SOM using input data, iteratively adjusting weights.

        Parameters:
        - input_data: np.ndarray, shape (n_samples, n_features)
            Input data for training.

        Returns:
        - np.ndarray
            The trained weights matrix of the SOM.
        """
        time_constant = self.max_iterations / np.log(self.neighborhood_radius)  # Decay constant        
        
        # Cache grid coordinates to speed up distance calculations.
        grid_x, grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing='ij')
        
        for iteration in range(self.max_iterations):
            # Decay neighborhood radius and learning rate.
            radius_decay = self.neighborhood_radius * np.exp(-iteration / time_constant)
            learning_rate_decay = self.learning_rate * np.exp(-iteration / time_constant)
            
            for sample in input_data:
                # Find the Best Matching Unit (BMU).
                bmu_index = np.argmin(np.sum((self.weights - sample) ** 2, axis=2))
                bmu_x, bmu_y = np.unravel_index(bmu_index, (self.width, self.height))
                
                # Define neighborhood boundaries.
                x_min = max(bmu_x - int(radius_decay) - 1, 0)
                x_max = min(bmu_x + int(radius_decay) + 2, self.width)
                y_min = max(bmu_y - int(radius_decay) - 1, 0)
                y_max = min(bmu_y + int(radius_decay) + 2, self.height)

                # Extract neighborhood coordinates.
                neighborhood_x = grid_x[x_min:x_max, y_min:y_max]
                neighborhood_y = grid_y[x_min:x_max, y_min:y_max]
                
                # Compute distances and influence from BMU.
                distances_sq = (neighborhood_x - bmu_x) ** 2 + (neighborhood_y - bmu_y) ** 2
                influence = np.exp(-(distances_sq) / (2 * (radius_decay ** 2)))
                
                # Update weights for neighborhood units.
                self.weights[x_min:x_max, y_min:y_max] += (
                    learning_rate_decay * influence[..., np.newaxis] * (sample - self.weights[x_min:x_max, y_min:y_max])
                )
        
        return self.weights
