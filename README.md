# Travelling Salesman Problem Art: What is the best algorithm to create art with one line

## Collaborators:
- Sofia Tkachenko, B22-DS-02
- Anastasiia Shvets, B22-DS-02
- Nika Lobanova, B22-GD-01

## Project description
This project is aimed to find the best way to draw TSP Art using metaheuristics inspired by nature, as well as examine possible ways to make TSP Art more colorful. We applied dithering to images from [CalTech 101](https://data.caltech.edu/records/mzrjq-6wc02) dataset with removed background, and then tried different nature inspired algorithms to solve TSP on given points. Afterwards, we added colors to the resulting images. 
The following methods were used for solving TSP Art:
- Ant Colony Optimization
- Artificial Bee Colony
- Genetic Algorithm
- Simulated Annealing

## Contents
### abc_alg.py
This file provides an impementation for Artifical Bee Colony metaheuristic to solve TSP and create TSP Art based on the solution.

### aco.py
This file provides an impementation for Ant Colony Optimization metaheuristic to solve TSP and create TSP Art based on the solution.

### dithering_and_colors_experiment.ipynb
This notebook shows how color can be added to TSP Art, as well as what dithering methods can be used to transform the original image into points.

### genetic_algorithm.py
This file provides an implementation for Genetic Algorithm metaheuristic to solve TSP and create TSP Art based on the solution.

### simulated_annealing.py
This file provides implementation for Simulated Annealing metaheuristic for solving TSP and creating TSP Art bsed on the solution.

## Comments
For now, the dithering algorithm from the [internet](https://github.com/MatthewMcGonagle/TSP_PictureMaker/blob/master/tsp_draw/dithering.py) is used.
