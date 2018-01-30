import numpy as np
import gmm_functions as gm
import pickle

# Set data directories
color_path_train = '/home/davideh29/Documents/UPenn/ESE 650/Image/'
bw_path_train = '/home/davideh29/Documents/UPenn/ESE 650/Label/'
# Load image and label data
color_images, bw_mask, barrel_distances, num_pixels = gm.load_training_data(color_path_train, bw_path_train)
# Train linear regression model
alpha_beta = gm.train_distance_regression(bw_mask, barrel_distances)
# Initial parameter estimates
num_mixtures = 2
mu = np.zeros([3, num_mixtures])  # Initial guess for mean of all mixtures
sigma = np.zeros([3, 3, num_mixtures])  # Initial guess for variance of all mixtures
weight = np.full(num_mixtures, (1.0 / num_mixtures))  # Initial guess for weighting of all mixtures
for i in range(0, num_mixtures):
    sigma[:, :, i] = np.eye(3) * 60 + i
    mu[:, i] = np.ones((1, 3)) * (120 + i)
# Train GMM
x_train = color_images # Get color images for training set
x_train = x_train[np.nonzero(bw_mask)] # Get rgb values for red pixels
mu, sigma, weight = gm.train_gmm(x_train, num_mixtures, mu, sigma, weight)
# Save model
model = [mu, sigma, weight, alpha_beta]
pickle.dump(model, open("gmm_model.p", "wb"))
