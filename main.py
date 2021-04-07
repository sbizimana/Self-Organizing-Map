import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
#np.random.seed(1)


class SOM:
    def __init__(self, size, dim, data):
        self.size = size
        self.dim = dim
        self.data = data
        self.sigma_naught = self.size - self.size*9/10
        # Initialize the weights to cover a portion of the data
        self.weights = np.random.rand(self.size, self.size, self.dim)
        self.winners = np.zeros((self.size, self.size))

    def train(self, epochs, plot=True):
        #tmp = np.reshape(self.weights, (self.size ** 2, 2))
        #lr_naught = 0.1
        #tau1 = 1000 / np.log(self.sigma_naught)
        #tau2 = 1000
        num_samples = len(self.data)
        total_iters = num_samples*epochs

        #self.data = np.random.rand(10000,3)*5

        # since lr should be start at 0.1 and never get below 0.01, maybe just linspace it through each epoch instead
        # of using tau2??? Do the same with sigma
        lr = np.linspace(0.9, 0.1, total_iters)
        sigma = np.linspace(self.sigma_naught, 0.1, total_iters)

        counter = 0
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            for index in indices:

                # calculate sigma and learning rate
                #sigma = sigma_naught * np.exp(-epoch/tau1)
                #lr = lr_naught * np.exp(-epoch/tau2)

                # grab random sample
                sample = self.data[index,:-1]

                # compete and adapt
                i, j = self.compete(sample)
                self.winners[i,j] += 1
                bmu = self.weights[i,j]
                self.adapt(lr[counter], sigma[counter], bmu, sample, i, j)

                if counter % 100 == 0:
                    print(f"Epoch: {epoch+1} / {epochs} \nIteration: {counter} / {total_iters}")
                counter += 1

                # Plot the SOM if you want to see every iteration
                if plot:
                    plot_data(self.data, self.calculate_edges(), self.weights, pause=True)

            # Plot the SOM avery every epoch
            plot_data(self.data, self.calculate_edges(), self.weights, pause=True if epoch != epochs - 1 else False, pause_time=1)

        edges = self.calculate_edges()
        return edges

    def compete(self, sample):
        best = [0, 0]
        dist = np.inf
        # find which neuron has the closest weight value to the sample
        for i in range(self.size):
            for j in range(self.size):
                d = euclidean_dist(self.weights[i][j], sample)
                if d < dist:
                    best = [i, j]
                    dist = d
        return best[0], best[1]

    def adapt(self, lr, sigma, bmu, sample, win_i, win_j):
        # Modify each neurons weight based on its distance to the winning neuron
        for i in range(self.size):
            for j in range(self.size):
                dist_to_winner = euclidean_dist(np.array([i,j]), np.array([win_i, win_j]))
                h = np.exp(-(dist_to_winner**2 / (2 * sigma)**2))
                delta = lr * h * (sample - self.weights[i,j])
                self.weights[i,j] += delta

    # Get the edges connecting nodes together
    def calculate_edges(self):
        edges = []
        # For each node, save its and it's neighbor's locations
        for i in range(self.size):
            for j in range(self.size):
                if i + 1 <= self.size-1:
                    start = self.weights[i, j]
                    end = self.weights[i + 1, j]
                    edges.append(np.array([start[0], end[0], start[1], end[1]]))

                if j + 1 <= self.size-1:
                    start = self.weights[i, j]
                    end = self.weights[i, j + 1]
                    edges.append(np.array([start[0], end[0], start[1], end[1]]))

        return np.array(edges)


def plot_data(data, edges, som, pause=False, pause_time=0.05):
    plt.scatter(data[:, 0], data[:, 1])
    if type(som) != np.ndarray:
        som = som.weights
    tmp = np.reshape(som, (som.shape[0] * som.shape[1], 2))
    for edge in edges:
        plt.plot(edge[0:2], edge[2:4], c='k')
    plt.scatter(tmp[:, 0], tmp[:, 1], c='r', s=10)
    plt.title(f"Self-Organizing Map on RBF Data \nLattice Size = {som.shape[0]}x{som.shape[1]} ")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(("RBF Data", "SOM"))
    if pause:
        plt.pause(pause_time)
        plt.clf()
    else:
        plt.show()


# Calculate edges, this is for when the SOM is loaded from memory as a numpy array.
def calculate_edges(som):
    edges = []
    if type(som) != np.ndarray:
        som = som.weights
    for i in range(som.shape[0]):
        for j in range(som.shape[0]):
            if i + 1 <= som.shape[0]-1:
                start = som[i, j]
                end = som[i + 1, j]
                edges.append(np.array([start[0], end[0], start[1], end[1]]))

            if j + 1 <= som.shape[0]-1:
                start = som[i, j]
                end = som[i, j + 1]
                edges.append(np.array([start[0], end[0], start[1], end[1]]))

    return np.array(edges)


# Construct the U-matrix. I'm very proud of this function :)
def construct_umatrix(som):
    # most_wins and red_scale are used for normalizing pixel values
    most_wins = np.max(som.winners)
    red_scale = np.linspace(0, 1, int(most_wins))
    size = som.weights.shape[0]

    # Calculate the maximum distance a node is from its neighbor. This is also used for normalizing pixel values.
    max_dist = 0
    for i in range(size):
        for j in range(size):
            cell = som.weights[i, j]
            if i - 1 >= 0:
                dist = euclidean_dist(cell, som.weights[i - 1, j])
                if dist >= max_dist:
                    max_dist = dist
            if i + 1 <= size - 1:
                dist = euclidean_dist(cell, som.weights[i + 1, j])
                if dist >= max_dist:
                    max_dist = dist
            if j - 1 >= 0:
                dist = euclidean_dist(cell, som.weights[i, j - 1])
                if dist >= max_dist:
                    max_dist = dist
            if j + 1 <= size - 1:
                dist = euclidean_dist(cell, som.weights[i, j + 1])
                if dist >= max_dist:
                    max_dist = dist

    # Initialize the U-matrix with zeros
    # Every node takes up a (10 x 10 x 3) space in the (size x size x 10 x 10 x 3) matrix,
    # this will later be converted to a (10*size x 10*size x 3) image.
    u_matrix = np.zeros((size, size, 10, 10, 3))
    # For every node...
    # 1. Use its number of wins to save its red-scale value for the u-matrix
    # 2. Calculate its distance to its neighbors for its grayscale fence value
    for i in range(size):
        for j in range(size):
            cell = som.weights[i,j]

            # Save the cell's red-scale value
            u_matrix[i, j, 1:9, 1:9, :] = np.ones((8, 8, 3)) * np.array([(red_scale[int(som.winners[i,j])-1]), 0, 0])

            # Save the cell's fence values
            if i-1 >= 0:
                dist = euclidean_dist(cell, som.weights[i-1,j])
                u_matrix[i, j, 0, 1:9, :] = np.ones(3) * (dist / max_dist)
            if i+1 <= size - 1:
                dist = euclidean_dist(cell, som.weights[i+1,j])
                u_matrix[i, j, 9, 1:9, :] = np.ones(3) * (dist / max_dist)
            if j-1 >= 0:
                dist = euclidean_dist(cell, som.weights[i,j-1])
                u_matrix[i, j, 1:9, 0, :] = np.ones(3) * (dist / max_dist)
            if j+1 <= size - 1:
                dist = euclidean_dist(cell, som.weights[i,j+1])
                u_matrix[i, j, 1:9, 9, :] = np.ones(3) * (dist / max_dist)

    # Turn the u-matrix into an image
    image = np.zeros((size*10, size*10, 3))
    u_i = 0
    u_j = 0
    for i in range(0, size*10, 10):
        for j in range(0, size*10, 10):
            image[i:i+10, j:j+10, :] = u_matrix[u_i, u_j, :, :, :]
            u_j += 1
        u_i += 1
        u_j = 0

    # plot the image!
    plt.imshow(image)
    plt.title(f"{size}x{size} U-Matrix")
    plt.show()


# Euclidean distance helper function
def euclidean_dist(a, b):
    return np.linalg.norm(a-b)


if __name__ == "__main__":
    data = np.loadtxt("data/RBF_data.csv",delimiter=',')

    '''
    CHANGE 'train' TO TRUE IF YOU WANT TO WATCH THE SOM TRAIN, OTHERWISE THE PRETRAINED SOM WILL BE LOADED.
    '''
    train = False
    if train:
        som = SOM(10, 2, data)
        edges = som.train(epochs=2, plot=False)
        with open("./models/SOM_new.pkl", 'wb') as f:
            pickle.dump(som, f)
    else:
        with open("./models/SOM.pkl", 'rb') as f:
            som = pickle.load(f)
        edges = calculate_edges(som)

    plot_data(data, edges, som)
    construct_umatrix(som)