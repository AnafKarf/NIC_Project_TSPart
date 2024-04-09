# Travelling Salesman Problem Art: What is the best algorithm to create art with one line

## Collaborators:
- Sofia Tkachenko, B22-DS-02
- Anastasiia Shvets, B22-DS-02
- Nika Lobanova, B22-GD-01

## Short description
This project is aimed to find the best way to draw TSP Art using metaheuristics inspired by nature, as well as examine possible ways to make TSP Art more colorful.

## Contents
### dithering_and_colors_experiment.ipynb
This notebook shows how color can be added to TSP Art, as well as what dithering methods can be used to transform the original image into points.

### genetic_algorithm.py
This file provides an implementation for Genetic Algorithm metaheuristic to solve TSP and create TSP Art based on the solution.

### simulated_annealing.py
This file provides implementation for Simulated Annealing metaheuristic for solving TSP and creating TSP Art bsed on the solution.

## Comments
For now, in genetic_algorithm.py and simulated_annealing.py the dithering algorithm from the [internet](https://github.com/MatthewMcGonagle/TSP_PictureMaker/blob/master/tsp_draw/dithering.py) is used.