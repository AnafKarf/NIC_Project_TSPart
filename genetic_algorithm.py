import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from PIL import Image

class DitheringMaker:
    '''
    Performs dithering on an image using the classical Floyd-Steinberg dithering algorithm. The
    dithering isn't applied to the last row or the edge columns. Instead the last row and edge
    columns are made all white as this will later not add any vertices when we extract vertices
    from the dithering.

    Members
    -------
    dithering : Numpy array of Int of Shape(n_rows, nCol)
        The array holding the dithering of the image. Note that the last row and edge columns will
        always be converted to white (i.e. 255).

    row_disp : Numpy array of Int of shape (2, 3)
        The row offsets to apply the diffusions to.

    col_disp : Numpy array of Int of shape (2, 3)
        The column offsets to apply the diffusions to.

    diffusion_prop : Numpy array of Float of shape (2, 3)
        The Floyd-Steinberg coefficients for diffusing the error in the
        dithering.
    '''

    def __init__(self):
        '''
        Initialize the dithering to None as we haven't performed any yet.

        The displacement of indices and the diffusion coefficients are for the classic
        Floyd-Steinberg dithering algorithm.
        '''

        self.dithering = None

        self.row_disp = np.full((2, 3), np.arange(0, 2)[:, np.newaxis], dtype = 'int')
        self.col_disp = np.full((2, 3), np.arange(-1, 2), dtype = 'int')
        self.diffusion_prop = np.array([[0, 0, 7],
                                        [3, 5, 1]]) / 16

    def make_dithering(self, pixels, cutoff = 255 / 2):
        '''
        Apply the classic Floyd-Steinberg dithering algorithm, but for simplicity
        we just make the edge columns and the last row all white (which will give
        us no vertices when we later extract vertices).

        Parameters
        ----------
        pixels : Numpy array of Int of shape (n_rows, n_cols)
            The gray scale pixels to apply the dithering to.

        cutoff : Float
            The cut off for making a dithering pixel either 0 or 255.

        Returns
        -------
        Numpy array of Int of Shape (n_rows, n_cols)
            The final dithering; each pixel is either 0 or 255 (black or white).
        '''

        # We use Floyd-Steinberg dithering. Error diffusion is
        # _     x     7/16
        # 3/16  5/16  1/16

        self.dithering = pixels.copy().astype('float')

        n_rows, n_cols = pixels.shape

        # Initialize the first column to be white.

        self.dithering[:][0] = 255

        # Iterate over each row, applying the dithering and diffusing the error.

        for row in range(n_rows - 1):
            for col in range(1, n_cols - 1):

                dither, error = self.get_dither(row, col, cutoff)
                self.dithering[row, col] = dither
                self.diffuse_error(error, row, col)

        # Make the last column and the last row all white.

        self.dithering[:, -1] = 255
        self.dithering[-1, :] = 255


        # Convert dithering to Numpy array of Int.

        self.dithering = self.dithering.astype('int')

        return self.dithering

    def get_dither(self, row, col, cutoff):
        '''
        Turn (dithered) pixel into either 0 or 255 using cutoff.

        Parameters
        ----------
        row : Int
            Index of pixel row.

        col : Int
            Index of pixel column.

        cutoff : Float
            The cutoff value to use for converting dithering value to either 0 or 255
            (black or white).

        Returns
        -------
        dither : Float
            Floating point value that is either 0.0 or 255.0 (black or white).

        error : Float
            The error in applying the conversion, this needs to be diffused to other pixels
            according to the dithering algorithm.
        '''

        pixel = self.dithering[row][col]

        if pixel < cutoff:
            dither = 0.0
        else:
            dither = 255.0

        error = pixel - dither

        return dither, error

    def diffuse_error(self, error, row, col):
        '''
        Diffuse the error from a (dithered) pixel conversion to black or white. The diffusion
        is applied to the neighbors of the pixel at position [row, col] according to the
        Floyd-Steinberg algorithm.

        Parameters
        ----------
        error : Float
            The size of error to diffuse to other pixels.

        row : Int
            The row index of where the conversion took place.

        col : Int
            The column index of where the conversion took place.
        '''

        self.dithering[row + self.row_disp, col + self.col_disp] += error * self.diffusion_prop

def get_vertices(dithering):
    '''
    Get the vertices from a black and white image, not grayscale (in particular a dithered image).
    Every black pixel (value 0.0) gives a vertex.

    Parameters
    ----------
    dithering : Numpy array of shape (n_rows, n_cols)
        The array of pixels for the dithered image.

    Returns
    -------
    Numpy array of shape (nVertices, 2)
        The xy-coordinates of the vertices.
    '''

    n_rows, n_cols = dithering.shape

    # Each black pixel gives a vertex.
    keep_pixel_mask = (dithering == 0)

    # Get the row and column indices of the vertices.

    rows = np.full(dithering.shape, np.arange(n_rows)[:, np.newaxis]).reshape(-1)
    cols = np.full(dithering.shape, np.arange(n_cols)).reshape(-1)

    rows = rows[keep_pixel_mask.reshape(-1)]
    cols = cols[keep_pixel_mask.reshape(-1)]

    # Get the xy-coordinate of the vertices. Make sure to transform row index so
    # that the last row has y value 0.

    vertices = np.stack([cols, n_rows - rows], axis = -1)

    return vertices


def getPixels(image, ds=1):
    '''
    Get the pixels as a numpy array from a PIL image.
    We can take the mean of each ds x ds subsquare as an array element inorder to down-size
    the size of the image if we want to.

    Parameters
    ----------
    image : PIL Image
        The PIL image to convert.

    ds : Int
        We take the mean of each ds x ds sub-square for a single element of our array.

    Returns
    -------
    2d Numpy array of floats
        The converted values of the pixels in the image. We use mean because we
        possibly took a mean over sub-squares.
    '''
    imwidth, imheight = image.size
    pixels = list(image.getdata())
    pixels = np.array(pixels).reshape((imheight, imwidth))
    pixels = [[pixels[i:i + ds, j:j + ds].mean() for j in np.arange(0, imwidth, ds)]
              for i in np.arange(0, imheight, ds)]
    return np.array(pixels)

class TSP_GA:
    def __init__(self, image_path, population_size=300, cx_pb=0.7, mut_pb=0.2, ngen=10, two_opt_percentage=0.3):
        self.image_path = image_path
        self.population_size = population_size
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.ngen = ngen
        self.two_opt_percentage = two_opt_percentage
        

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Initialize for DEAP 
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", self.swap_mutation, indpb=0.05)
        self.toolbox.register("select", tools.selRoulette)

    def process_image(self):
        '''
        Process the input image for TSP solving.
        '''
        image = Image.open(self.image_path).convert('L')
        pixel_image = getPixels(image, ds=3)
        ditherer = DitheringMaker()
        dithered_image = ditherer.make_dithering(pixel_image)
        self.vertices = get_vertices(dithered_image)
        self.distance_matrix = self.calculate_distance_matrix(self.vertices)

        self.toolbox.register("indices", random.sample, range(len(self.vertices)), len(self.vertices))
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.total_distance)

    def calculate_distance_matrix(self, vertices):
        '''
        Calculate the distance matrix between vertices.
        '''
        return np.sqrt(((np.array(vertices)[:, np.newaxis, :] - np.array(vertices)[np.newaxis, :, :]) ** 2).sum(axis=2))

    def total_distance(self, individual):
        '''
        Calculate the total distance of a given individual's path.
        '''
        return sum(self.distance_matrix[individual[i-1], individual[i]] for i in range(len(individual))),

    def swap_mutation(self, individual, indpb):
        '''
        Perform a swap mutation on an individual.

        Parameters
        ----------
        individual : list
            The individual to mutate.

        indpb : float
            The probability of mutation for each gene.

        Returns
        -------
        tuple
            A tuple containing the mutated individual.
        '''
        for i in range(len(individual)):
            if random.random() < indpb:
                swap_idx = random.randint(0, len(individual)-1)
                individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
        return individual,

    def apply_two_opt(self, population, percentage=0.2):
        '''
        Apply 2-opt optimization to a percentage of the population.
        '''
        num_individuals = int(len(population) * percentage)
        selected_indices = random.sample(range(len(population)), num_individuals)
    
        for idx in selected_indices:
            ind = population[idx]
            improved = True
            while improved:
                improved = False
                for i in range(1, len(ind) - 1):
                    for j in range(i + 1, len(ind)):
                        if j-i == 1: continue  # Skip adjacent nodes
                        if self.distance_matrix[ind[i-1]][ind[i]] + self.distance_matrix[ind[j-1]][ind[j]] > self.distance_matrix[ind[i-1]][ind[j-1]] + self.distance_matrix[ind[i]][ind[j]]:
                            ind[i:j] = ind[i:j][::-1]  # Reverse the segment
                            improved = True
            ind.fitness.values = self.total_distance(ind)

    def solve(self):
        self.process_image()
        population = self.toolbox.population(n=self.population_size)

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)

        for gen in range(self.ngen):
            offspring = algorithms.varAnd(population, self.toolbox, self.cx_pb, self.mut_pb)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population = self.toolbox.select(offspring, len(population))
            self.apply_two_opt(population)

        best_ind = tools.selBest(population, 1)[0]
        print("Best individual is: %s\nWith fitness: %s" % (best_ind, best_ind.fitness.values))
        self.plot_path(best_ind)

    def plot_path(self, best_ind):
        '''
        Plot the best path found by the solver.
        '''
        best_path = [self.vertices[i] for i in best_ind] + [self.vertices[best_ind[0]]]
        x, y = zip(*best_path)
        plt.figure(figsize=(10, 10))
        plt.plot(x, y, '-o', markersize=3, linewidth=1)
        plt.title("Best Path")
        plt.show()

if __name__ == "__main__":
    solver = TSP_GA("flower.png")
    solver.solve()
