import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
from scipy.special import logsumexp
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage import measure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def load_training_data(color_path, bw_path):
    # Get file names
    color_files = [f for f in listdir(color_path) if isfile(join(color_path, f))]
    bw_files = [f for f in listdir(bw_path) if isfile(join(bw_path, f))]
    # Sort to make sure color images and masks line up
    color_files.sort()
    bw_files.sort()
    # List of image red pixel values
    color_images = []
    bw_mask = []
    barrel_distances = []
    num_pixels = 0
    # Loop through images
    for i in range(0, len(color_files)):
        # Get barrel distance(s)
        dist = color_files[i].split(".")
        dist = dist[0].split("_")
        barrel_distances.append(dist)
        # Load color image
        hsv = cv2.cvtColor(cv2.imread(color_path + color_files[i], 1), cv2.COLOR_BGR2HSV)
        color_images.append(hsv)
        # Place mask indices in array
        bw_mask.append(cv2.imread(bw_path + bw_files[i], 0))
        num_pixels += len(np.nonzero(bw_mask[i])[0])
    # Convert to ndarray and return
    color_images = np.array(color_images)
    bw_mask = np.array(bw_mask)
    barrel_distances = np.array(barrel_distances)
    return color_images, bw_mask, barrel_distances, num_pixels


def load_testing_data(color_path):
    # Get file names
    color_files = [f for f in listdir(color_path) if isfile(join(color_path, f))]
    # List of images
    color_images = []
    # Loop through images
    for i in range(0, len(color_files)):
        # Load color image
        hsv = cv2.cvtColor(cv2.imread(color_path + color_files[i], 1), cv2.COLOR_BGR2HSV)
        color_images.append(hsv)
    # Convert to ndarray and return
    color_images = np.array(color_images)
    return color_images, color_files


def train_distance_regression(bw_mask, barrel_distances):
    print "Training regression model"
    # Get height and width values from training array
    height_train = []
    width_train = []
    dist_train = []
    for i in range(0, len(bw_mask)):
        # Get components/bounding boxes of binary images
        component_temp, reg_temp = get_components(bw_mask[i], False)
        temp_dist = np.array(barrel_distances[i]).astype(float)
        temp_height = []
        temp_width = []
        for b in range(0, len(reg_temp)):
            minr, minc, maxr, maxc = reg_temp[b].bbox
            temp_height.append(maxr - minr)
            temp_width.append(maxc - minc)
        # Sort by height and match to sorted distances
        temp_height = np.array(temp_height)
        temp_width = np.array(temp_width)
        sort_ind = np.argsort(temp_height)
        sort_dist = temp_dist.argsort()[::-1]
        if len(height_train) == 0:
            height_train.append(temp_height[sort_ind])
            width_train.append(temp_width[sort_ind])
            dist_train.append(temp_dist[sort_dist])
            height_train = np.array(height_train)
            width_train = np.array(width_train)
            dist_train = np.array(dist_train)
        else:
            height_train = np.append(height_train, temp_height[sort_ind])
            width_train = np.append(width_train, temp_width[sort_ind])
            dist_train = np.append(dist_train, temp_dist[sort_dist])
    # Calculate alpha and beta coefficients to predict distance through linear regression
    A = np.column_stack([height_train, width_train])
    b = np.reciprocal(dist_train)
    alpha_beta = np.linalg.lstsq(A, b)
    return alpha_beta


def get_components(img, disp):
    # Apply binary erosion filter and morphological dilation
    img = binary_erosion(img, structure=np.ones((10, 10)))
    img = binary_dilation(img, structure=np.ones((30, 30)))
    components = measure.label(img)
    regions = []
    if disp:
        fig, ax = plt.subplots(1)  # Create figure and axes
        ax.imshow(img)  # Display image
    for region in measure.regionprops(components):  # Find connected components
        if region.area > 2000:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            if (maxc - minc) < (maxr - minr):
                regions.append(region)
                # Display images to compare
                if disp:
                    plt.scatter(region.centroid[1], region.centroid[0])
                    ax.add_patch(rect)
    # Display image and components
    if disp:
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    regions = np.array(regions)
    return components, regions


def get_components_color(img, color_img, disp):
    # Apply binary erosion filter and morphological dilation
    img = binary_erosion(img, structure=np.ones((10, 10)))
    img = binary_dilation(img, structure=np.ones((30, 30)))
    components = measure.label(img)
    regions = []
    color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2RGB)
    if disp:
        fig, ax = plt.subplots(1)  # Create figure and axes
        ax.imshow(color_img)  # Display image
    for region in measure.regionprops(components):  # Find connected components
        if region.area > 2000:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            if (maxc - minc) < (maxr - minr):
                regions.append(region)
                # Display images to compare
                if disp:
                    plt.scatter(region.centroid[1], region.centroid[0])
                    ax.add_patch(rect)
    # Display image and components
    if disp:
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    regions = np.array(regions)
    return components, regions


def log_phi(x, mu, sigma):
    temp = np.dot(x - mu, np.linalg.inv(sigma))
    coeff = np.sqrt(2.0 * (np.pi ** len(x[1])) * np.linalg.det(sigma))
    out = np.subtract(-0.5 * np.sum(np.multiply(temp, np.subtract(x, mu)), axis=1), np.log(coeff))
    return out


def update_sigma(x, mu, sigma, r_k_x):
    r_tile = np.multiply(np.tile(r_k_x, (3, 1)), np.transpose(x - mu))
    outer_x = np.dot(r_tile, x - mu)
    sigma = np.divide(outer_x, sum(r_k_x))
    return sigma


def train_gmm(x, num_mixtures, mu, sigma, weight):
    print "Training GMM"
    # Initialize variables
    change_likelihood = float('inf')  # Change in log likelihood between each iteration
    alpha = 5  # Stop condition for EM
    r_k_x = np.empty([len(x), num_mixtures])  # Vector for expectation of all data for each class
    log_likelihood_new = 0
    log_likelihood_prev = 0

    # Loop and iterate between E and M step
    while change_likelihood > alpha:

        # E-STEP: Calculate expectations using current params
        # Calculate phi for all data points for each class
        for i in range(0, num_mixtures):
            r_k_x[:, i] = log_phi(x, mu[:, i], sigma[:, :, i])
        # Log of denominator for update of r_k_x
        log_denom = logsumexp((np.log(weight) + r_k_x), axis=1)
        # Update expectation r_k_x
        for i in range(0, num_mixtures):
            r_k_x[:, i] = np.exp(np.log(weight[i]) + r_k_x[:, i] - log_denom)

        # M-STEP: Update parameters using calculated expectations
        for i in range(0, num_mixtures):
            sigma[:, :, i] = update_sigma(x, mu[:, i], sigma[:, :, i], r_k_x[:, i])
            mu[:, i] = np.sum(np.multiply(np.tile(r_k_x[:, i], (3, 1)), np.transpose(x)), axis=1) / np.sum(r_k_x[:, i])
            weight[i] = np.sum(r_k_x[:, i]) / len(x)

        # Calculate log likelihood
        log_likelihood_prev = log_likelihood_new
        log_likelihood_new = np.sum(log_denom)
        if log_likelihood_prev != 0:
            change_likelihood = log_likelihood_new - log_likelihood_prev
        # print "Mu: " + format(mu)
        # print "Cov: "+format(sigma)
        # print "Weights: " + format(weight)
        print "Log Likelihood: " + format(log_likelihood_new)

    return mu, sigma, weight


def get_prob_samples(x, mu, sigma, weight, cutoff_prob):
    num_mixtures = mu.shape[1]
    out = np.empty([x.shape[0], x.shape[1], x.shape[2]], dtype=np.uint8)
    # Get phi for each pixel in image
    components = []
    regions = []
    for i in range(0, x.shape[0]):
        # Get rgb pixels for one image
        rgb_x = np.reshape(x[i], [x.shape[1] * x.shape[2], 3])
        # Get log prob values for image
        log_phi_x = np.empty([len(rgb_x), num_mixtures])
        for mix in range(0, num_mixtures):
            log_phi_x = log_phi(rgb_x, mu[:, mix], sigma[:, :, mix])
        # Use cutoff probability to get red pixels
        # Reshape back into image
        out[i] = (np.reshape(log_phi_x, [x.shape[1], x.shape[2]]) > cutoff_prob).astype(np.uint8)
        component_temp, reg_temp = get_components_color(out[i], x[i], True)
        regions.append(reg_temp)
        components.append(component_temp)
    regions = np.array(regions)
    components = np.array(components)
    return out, components, regions


def compute_distances(regions_test, alpha_beta):
    dist_test = []
    for d in range(0, len(regions_test)):
        temp_height_width = []
        for t in range(0, len(regions_test[d])):
            minr, minc, maxr, maxc = regions_test[d][t].bbox
            temp_height_width.append([maxr - minr, maxc - minc])
        temp_height_width = np.array(temp_height_width)
        dist_test.append(np.reciprocal(np.dot(temp_height_width, alpha_beta[0])))
    return dist_test

