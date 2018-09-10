# Assignment-for-Evolutionary-and-Heuristic-Optimisation
This is a programming assignment (on Python) that implements 4 search algorithms (random search, hill-climber, iterated local search and evolutionary algorithm) to find the permutation with an objective function which minimises the sum of the distance between adjacent elements of the permutation. 

IMPORTANT! In order to read the colour.txt file we need to delete the first lines with the comments in the file. Specifically, we neeed to delete:
# This file contains rows of three floating point values.
# Each row represents a colour by its red, green, blue coordinates.
# The first uncommented line is the number of colours.
The file should contain only numbers. (the number of colours and the number that constitute every colour)
I explain how every operator works in the assignment and I put comments in code.
In the 9th line of the code there is the variable size which sets the number of colours that we examine. (so we have to change it to 10, 100 and 1000 for this assignment)
In the beginning of the code there are functions and after that I print the results of every algorithm that we want to extract (after the 20 runs).
First we read the data from the file (n, colours = read_kfile('colours.txt')) and we intialise a list with random colour indexes (rand=initialise(size))
If we want to print the result of every algorithm isolated (in order to understand how much time does every algorithm to compute the results) we have to comment to the the other algorithms.
In the end a line graph is printed that compared the four algorithm.
