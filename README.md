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
### TSP_algorithms
This directory contains implementations of different metaheuristics for solving Travelling Salesman Problem
- abc_alg.py - Artificial Bee Colony metaheuristic, by Nika Lobanova
- aco.py - Ant Colony Optimization metaheuristic, by Anastasiia Shvets
- genetic_algorithm.py - Genetic Algorithm metaheuristic, by Nika Lobanova
- simulated_annealing.py - Simulated Annealing metaheuristic, by Anastasiia Shvets

### experiments
This directory contains IPYNB files with experiments conducted during the project.
- ABC_algorithm.ipynb, ACO.ipynb, GA_algorithm.ipynb, Simulated_Annealing.ipynb - experiments measuring the time it takes for different metaheuristic to solve the same problem (transform image of a lotus into TSP Art), by Nika Lobanova and Anastasiia Shvets
- dithering_and_colors_experiment.ipynb - experiments with different dithering algorithms, grayscale images, background removal and techniques of adding color to TSP Art, by Sofia Tkachenko

### images
This directory contains images used during the experiments.

### main.py
This file contains the best metaheuristic (Genetic Algorithm) and shows how the project works overall.

### requirements.txt
File containing python packages needed for the project to work.
