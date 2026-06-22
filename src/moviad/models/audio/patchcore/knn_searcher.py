from sklearn import random_projection
from .kcenter_greedy import KCenterGreedy

class KNNSearcher:
    """
    A class for k-NN search with dimention resuction (random projection)
    and subsampling (k-center greedy method) features.
    """
    def __init__(self, projection=True, subsampling=True, sampling_ratio=0.01):
        """
        Constructor of the KNNSearcher class.

        Args:
            projection     (bool) : Enable random projection if true.
            subsampling    (bool) : Enable subsampling if true.
            sampling_ratio (float): Ratio of subsampling.
        """
        self.projection     = projection
        self.subsampling    = subsampling
        self.sampling_ratio = sampling_ratio

    def fit(self, x):
        """
        Train k-NN search model.

        Args:
            x (torch.Tensor): Training data of shape (n_samples, n_features).
        """
        # Apply random projection if specified. Random projection is used for reducing
        # dimention while keeping topology. It makes the k-center greedy algorithm faster.
        if self.projection:

            print("Sparse random projection")

            # If number of features is much smaller than the number of samples, random
            # projection will fail due to the lack of number of features. In that case,
            # please increase the parameter `eps`, or just skip the random projection.
            projector = random_projection.SparseRandomProjection(n_components="auto", eps=0.90)
            projector.fit(x.cpu().numpy())

            # Print the shape of random matrix: (n_features_after, n_features_before).
            shape = projector.components_.shape
            print(f"Shape of the random matrix {str(shape)}")

        # Set None if random projection is no specified.
        else: 
            projector = None

        # Execute coreset subsampling.
        if self.subsampling:
            print("Coreset subsampling")
            n_select = int(x.shape[0] * self.sampling_ratio)
            selector = KCenterGreedy(x)
            indices  = selector.coreset_selection(projector, n_select)
            centers = x[indices, :]

        

        