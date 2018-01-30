import pickle
import gmm_functions as gm

color_path_test = '/home/davideh29/Documents/UPenn/ESE 650/Test/'
# Load GMM
model = pickle.load(open("gmm_model.p", "rb"))
mu = model[0]
sigma = model[1]
weight = model[2]
alpha_beta = model[3]
test_images, color_files = gm.load_testing_data(color_path_test)
# Test on test data
out, components_test, regions_test = gm.get_prob_samples(test_images, mu, sigma, weight, -14.4)
dist_test = gm.compute_distances(regions_test, alpha_beta)
# Print centroids and distances
for i in range(0, len(dist_test)):
    print "Image: " + color_files[i]
    print "Distances: "
    for d in range(0, len(dist_test[i])):
        print "Barrel #" + format(d+1)
        print "\tDistance: " + format(dist_test[i][d])
        minr, minc, maxr, maxc = regions_test[i][d].bbox
        height = maxr - minr
        width = maxc - minc
        print "\tCentroid: " + format([regions_test[i][d].centroid[1], regions_test[i][d].centroid[0]])
        print "\tHeight: " + format(height)
        print "\tWidth: " + format(width)
    print ""
