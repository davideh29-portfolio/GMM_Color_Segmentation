Files:

barrels_train.py
- Generates model to be loaded by barrels_test.py
- Specify directories for training images and labels at the start of the file as color_path and bw_path
- Corresponding files in either folder should have the same name
- Saves model as gmm_model.p

barrels_test.py
- Runs the model on a testing set and prints centroids, heights and widths for each barrel in each image
- Specify directories for test images at the start of the file as color_path_test
- Displays images with bounding boxes
- Prints the result in the console

barrels.py
- Runs cross-validation on provided image and labels
- Specify directories for image and labels at the start of the file as color_path and bw_path
- Corresponding files in either folder should have the same name

gmm_functions.py
- Contains all the functions used in the other three files for training the model and classifying pixels
- Change the value of the second input to the function call of get_components_color() to "True" inside the function definition of train_distance_regression() to display bounding boxes on color images when running barrels_test.py
