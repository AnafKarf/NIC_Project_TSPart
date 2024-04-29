# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os
import cv2
import numpy as np
from PIL import Image
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import pandas as pd
from rembg import remove


def load_dataset():
    """
    The `load_dataset` function downloads and extracts the Caltech 101 dataset, then creates a dataframe
    containing paths to all image files in the dataset.
    :return: The function `load_dataset()` returns a pandas DataFrame containing the paths to all the
    image files (with the extension .jpg) found in the subfolders of the directory
    "./101_ObjectCategories".
    """
    # load Caltech 101 database from https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1

    os.system("wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip")
    os.system("unzip caltech-101.zip")

    os.system("tar -xf ./caltech-101/101_ObjectCategories.tar.gz")

    # parse directory /101_ObjectCategories into a dataframe of images, in all subfolders

    image_paths = []
    for root, dirs, files in os.walk("./101_ObjectCategories"):
        for file in files:
            if file.endswith(".jpg"):
               image_paths.append(os.path.join(root, file))

    df_images = pd.DataFrame({"path": image_paths})
    return df_images


def remove_background(path):
    """
    The function `remove_background` takes an image file path as input, removes the background from the
    image, fills the background with white color, resizes the image, and returns the modified image.
    
    :param path: The `path` parameter in the `remove_background` function is a string that represents
    the file path to an image file. This function seems to be designed to remove the background from an
    image using OpenCV and then resize the resulting image
    :return: the variable `output`, which is the result of processing the input image to remove the
    background and then resizing it.
    """
    input = Image.open(path)
    pil_output = remove(input)
    output = np.asarray(pil_output)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cv2.floodFill(output, None, (0, 0), (255, 255, 255))
    img = cv2.resize(output, (0,0), fx = 0.5, fy = 0.5)
    return img


class DitheringMaker:
    """
    Class for applying Floyd-Steinberg dithering to images.

    Floyd-Steinberg dithering is a technique for converting grayscale images
    to black and white while preserving detail by diffusing quantization errors
    to neighboring pixels.
    """

    def __init__(self):
        """
        Initializes the DitheringMaker with the error diffusion proportions.
        """
        # Error diffusion proportions for Floyd-Steinberg dithering
        # Arrangement:
        # _     x     7/16
        # 3/16  5/16  1/16
        self.diffusion_prop = np.array([[0, 0, 7], [3, 5, 1]]) / 16


    def make_dithering(self, pixels, cutoff=255 / 2):
        """
        Applies Floyd-Steinberg dithering to the input image.

        Args:
            pixels (np.array): The input image as a NumPy array.
            cutoff (float, optional): The threshold for determining black/white pixels. Defaults to 127.5.

        Returns:
            np.array: The dithered image as a NumPy array.
        """

        # Create a working copy of the image as float for calculations
        dithering = pixels.copy().astype('float')
        n_rows, n_cols = pixels.shape

        # Initialize the first column to be white (assuming background color)
        dithering[:, 0] = 255

        # Pre-calculate row and column offsets for error diffusion
        row_disp = np.full((2, 3), np.arange(0, 2)[:, np.newaxis], dtype='int')
        col_disp = np.full((2, 3), np.arange(-1, 2), dtype='int')

        # Iterate over each row (except the last) and column (except first and last)
        for row in range(n_rows - 1):
            for col in range(1, n_cols - 1):
                old_pixel = dithering[row, col]

                # Apply dithering: set pixel to black or white based on cutoff
                new_pixel = 255.0 if old_pixel > cutoff else 0.0
                dithering[row, col] = new_pixel

                # Calculate the quantization error
                error = old_pixel - new_pixel

                # Distribute the error to neighboring pixels using diffusion proportions
                dithering[row + row_disp, col + col_disp] += error * self.diffusion_prop

        # Make the last column and the last row all white (border pixels)
        dithering[:, -1] = 255
        dithering[-1, :] = 255

        # Convert the dithered image back to integers (black/white)
        return dithering.astype('int')


def get_vertices(dithering):
    """
    Extracts vertices from a black and white (dithered) image.

    Args:
        dithering (np.array): The dithered image as a NumPy array.

    Returns:
        np.array: An array of shape (n_vertices, 2) containing the xy-coordinates of the vertices.
    """

    n_rows, n_cols = dithering.shape
    keep_pixel_mask = (dithering == 0)  # Mask for black pixels (vertices)

    rows, cols = np.mgrid[:n_rows, :n_cols]  # Create row and column indices
    rows, cols = rows[keep_pixel_mask], cols[keep_pixel_mask]  # Filter by mask

    vertices = np.stack([cols, n_rows - 1 - rows], axis=-1)  # Create vertex coordinates (invert y-axis)
    return vertices


def get_pixels(image, downsample_factor=1):
    """
    Converts a PIL image to a NumPy array with optional downsampling.

    Args:
        image (PIL.Image): The PIL image to convert.
        downsample_factor (int, optional): Downsampling factor. Defaults to 1 (no downsampling).

    Returns:
        np.array: A 2D NumPy array of pixel values (mean of downsampled regions if applicable).
    """

    ds = downsample_factor
    imwidth, imheight = image.size
    pixels = np.array(list(image.getdata())).reshape((imheight, imwidth))  # Get pixel data as NumPy array

    # Downsample by taking the mean of ds x ds sub-squares
    downsampled_pixels = np.array([[pixels[i:i+ds, j:j+ds].mean()
                                    for j in range(0, imwidth, ds)]
                                   for i in range(0, imheight, ds)])
    return downsampled_pixels


# This Python class implements a Genetic Algorithm for solving the Traveling Salesman Problem (TSP)
# using DEAP, with methods for processing images, optimizing paths, and plotting the results.
class TSP_GA:
    def __init__(self, image_path, population_size=200, ngen=10):
        """
        The function initializes parameters for an evolutionary algorithm with a specified image path,
        population size, and number of generations.

        :param image_path: The `image_path` parameter in the `__init__` method is a string that
        represents the path to an image file. This path will be used by the class or function to load
        and process the image data
        :param population_size: The `population_size` parameter represents the number of individuals in
        the population for a genetic algorithm. In this context, it specifies the size of the population
        used in the genetic algorithm optimization process for the given image path, defaults to 200
        (optional)
        :param ngen: The `ngen` parameter represents the number of generations
        for which the genetic algorithm will run. It determines how many iterations or generations of
        the genetic algorithm will be executed to evolve the population towards a solution, defaults to
        10 (optional)
        """
        self.image_path = image_path
        self.population_size = population_size
        self.ngen = ngen
        self.toolbox = base.Toolbox()
        self.best_paths = []
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)


    def evaluate(self, individual):
        """
        The function evaluates the total length of a path based on a given individual's sequence.

        :param individual: The `individual` parameter in the `evaluate` function represents a specific
        path or route in a traveling salesman problem. It is a list of nodes representing the order in
        which the salesman visits the cities. The function calculates the total length of the path by
        summing the distances between consecutive nodes in the `
        :return: A tuple containing the total length of the path is being returned.
        """
        distance_matrix = self.distance_matrix
        tour_length = sum(distance_matrix[individual[i - 1], individual[i]] for i in range(len(individual)))
        return (tour_length,)


    def calculate_distance_matrix(self, vertices):
        """
        The function calculates the Euclidean distance matrix between given vertices.

        :param vertices: The `vertices` parameter is a numpy array containing the coordinates of the
        vertices in a multidimensional space. Each row of the array represents the coordinates of a
        single vertex. The function `calculate_distance_matrix` calculates the Euclidean distance
        between each pair of vertices and returns a matrix where the element at position
        :return: The function `calculate_distance_matrix` returns the Euclidean distance matrix between
        the input vertices.
        """
        return np.sqrt(((vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]) ** 2).sum(axis=2))


    def two_opt(self, individual):
        """
        The `two_opt` function implements a 2-opt optimization algorithm to improve a given individual's
        path based on a distance matrix.

        :param individual: The `individual` parameter in the `two_opt` function represents a list that
        contains the order of visiting nodes in a tour. Each element in the list represents a node, and
        the order of the nodes determines the sequence of visiting them in the tour. The function
        applies a 2-opt optimization technique
        :return: The `two_opt` function is returning the optimized individual after applying the 2-opt
        optimization technique. The optimized individual is returned as a tuple containing the
        individual itself.
        """
        improved = True
        while improved:
            improved = False
            for i in range(1, len(individual) - 2):
                for j in range(i + 2, len(individual) - 1):
                    if self.distance_matrix[individual[i - 1]][individual[i]] + \
                        self.distance_matrix[individual[j]][individual[j + 1]] > \
                        self.distance_matrix[individual[i - 1]][individual[j]] + \
                        self.distance_matrix[individual[i]][individual[j + 1]]:
                        individual[i:j + 1] = reversed(individual[i:j + 1])
                        improved = True
        return individual,


    def setup_toolbox(self):
        """
        Sets up a DEAP toolbox for evolutionary algorithms.
        """
        self.toolbox.register("indices", random.sample, range(len(self.vertices)), len(self.vertices))
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", self.two_opt)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate)


    def process_image(self, image):
        """
        The `process_image` function processes an input image for TSP solving by downsampling, dithering,
        extracting vertices, calculating distance matrix, and setting up a toolbox.

        :param image: The `image` parameter is the input image that will be processed for solving the
        Traveling Salesman Problem (TSP). The image will be used to extract pixels, create a dithered
        version of the image, obtain vertices from the dithered image, and calculate a distance matrix based
        on
        """
        pixels = get_pixels(image, downsample_factor=3)
        ditherer = DitheringMaker()
        dithered_image = ditherer.make_dithering(pixels)
        self.vertices = get_vertices(dithered_image)
        self.distance_matrix = self.calculate_distance_matrix(self.vertices)
        self.setup_toolbox()


    def ga_algorithm(self):
        """
        The function `ga_algorithm` implements a genetic algorithm using the DEAP library to
        optimize a population of individuals.
        :return: The code is returning the best individual found by the genetic algorithm after running the
        evolutionary algorithm using the `eaSimple` function. The best individual is stored in the
        HallOfFame object `hof`, and the code returns this best individual by accessing it using
        `hof.items[0]`.
        """
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.average)
        algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.2, ngen=self.ngen, stats=stats, halloffame=hof, verbose=True)
        return hof.items[0]


    def solve(self):
        """
        This function opens an image, processes it, runs a genetic algorithm to find the best solution, and
        then plots the path of the best solution.
        """
        image = Image.open(self.image_path).convert('L')
        self.process_image(image)
        best_solution = self.ga_algorithm()
        self.plot_path(best_solution)


    def solve_color(self):
        """
        This function solves the Traveling Salesman Problem (TSP) for color images by splitting the
        image into its RGB channels, processing each channel separately, applying a genetic algorithm to
        find the best path, and plotting the paths.
        """
        b, g, r = cv2.split(cv2.imread(self.image_path))
        b_pil = Image.fromarray(b)
        g_pil = Image.fromarray(g)
        r_pil = Image.fromarray(r)

        counter = 0
        colors = ['blue', 'green', 'red']

        for image in [b_pil, g_pil, r_pil]:
            print('Solving TSP for', colors[counter])
            self.process_image(image)
            best_solution = self.ga_algorithm()
            self.best_paths.append([self.vertices[i] for i in best_solution] + [self.vertices[best_solution[0]]])
            self.plot_path(best_solution)
            counter += 1

        self.plot_path_color(self.best_paths)


    def plot_path(self, best_solution):
        '''
        Plot the best path found by the solver.
        '''
        best_path = [self.vertices[i] for i in best_solution] + [self.vertices[best_solution[0]]]
        x, y = zip(*best_path)
        plt.figure()
        plt.plot(x, y, '-o', markersize=2, linewidth=1)
        plt.title("Best Path")
        plt.show()


    def plot_path_color(self, best_paths):
        """
        The function `plot_path_color` plots three paths in different colors on a black background.

        :param best_paths: Solutions for TSPs of each color respectively.
        """
        x_b, y_b = zip(*best_paths[0])

        x_g, y_g = zip(*best_paths[1])
        x_r, y_r = zip(*best_paths[2])

        fig, ax = plt.subplots()
        ax.set_facecolor('black')

        # Plotting all the curves simultaneously
        ax.plot(x_r, y_r, '-o', markersize=2, linewidth=1, alpha=0.8,
                 color='cyan')  # inverse for red
        ax.plot(x_g, y_g, '-o', markersize=2, linewidth=1, alpha=0.8,
                 color='magenta') # inverse for green
        ax.plot(x_b, y_b, '-o', markersize=2, linewidth=1, alpha=0.8,
                 color='yellow') # inverse for blue

        plt.title("Colorful TSP art")
        plt.show()
     

# We chose GA due to it's outstanding performance in our experiments
solver = TSP_GA("images/small_lotus.png")
solver.solve_color()
