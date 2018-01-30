import numpy as np
import gmm_functions as gm

color_path = '/home/davideh29/Documents/UPenn/ESE 650/Image/'
bw_path = '/home/davideh29/Documents/UPenn/ESE 650/Label/'

color_images, bw_mask, barrel_distances, num_pixels = gm.load_training_data(color_path, bw_path)

# Cross-validation of GMM and distance predictions
num_xval = 4
size_test = len(bw_mask) / num_xval
# Loop through number of xvals
for xval in range(0, num_xval):
    # Get random partition
    rand_indices = np.arange(len(bw_mask))
    np.random.shuffle(rand_indices)
    train_indices = rand_indices[size_test:]
    test_indices = rand_indices[0:size_test]
    # Train regression model for distance
    alpha_beta = gm.train_distance_regression(bw_mask[train_indices], barrel_distances[train_indices])
    # Initial parameter estimates
    num_mixtures = 2
    mu = np.zeros([3, num_mixtures])  # Initial guess for mean of all mixtures
    sigma = np.zeros([3, 3, num_mixtures])  # Initial guess for variance of all mixtures
    weight = np.full(num_mixtures, (1.0 / num_mixtures))  # Initial guess for weighting of all mixtures
    for i in range(0, num_mixtures):
        sigma[:, :, i] = np.eye(3) * 60 + i
        mu[:, i] = np.ones((1, 3)) * (120 + i)
    # Train gaussian mixture model on training data
    x_train = color_images[train_indices] # Get color images for training set
    x_train = x_train[np.nonzero(bw_mask[train_indices])] # Get rgb values for red pixels
    mu, sigma, weight = gm.train_gmm(x_train, num_mixtures, mu, sigma, weight)
    # Test model and calculate accuracy
    x_test = color_images[test_indices] # Get testing images
    out, components_test, regions_test = gm.get_prob_samples(x_test, mu, sigma, weight, -14.4)
    # Loop through test sample regions and compute distances
    dist_test = gm.compute_distances(regions_test, alpha_beta)
    dist_test = np.hstack(np.array(dist_test))
    dist_test_actual = np.hstack(np.array(barrel_distances[test_indices])).astype(float)
    if len(dist_test) != len(dist_test_actual):
        print "Had false detection"
    else:
        mean_error_dist = np.sqrt(np.mean(np.square(np.subtract(dist_test, dist_test_actual))))
        print "Mean squared error distance: " + format(mean_error_dist)

print "fin"
