## RRT Challenges
**Michael Wiznitzer**

Northwestern University MSR Hackathon (Fall 2017)

## Introduction
#### Objective
To complete 3 challenges that implement the [RRT algorithm](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree) in various ways. The 3 challenges are:
 1. To create a simple tree with _N_ number of nodes starting from the center of a 100x100 plot
 2. To find a path from _Point A_ to _Point B_ in a plot with 20 randomly generated circles used as obstacles
 3. To find a path from _Point A_ to _point B_ in a binary image of the Northwestern "N" symbol shown below

 ![N_map](imgs/N_map.png)

Simply put, the RRT algorithm works by:
 - starting at an inital node
 - choosing some other random node
 - finding the closest node in the tree and connecting the random node to that node
 - repeating the process again for _N_ nodes

In the common case that there are obstacles, then it is imperitave to first check to see if the random node can even connect to the closest node in the tree.
For the last 2 challenges which involved this validation, [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) was used to check if any of the points on the line connecting these 2 nodes were part of obstacles or not.

#### Simple RRT Tree
A tree created with a 100 nodes is shown below

![simple_rrt.png](imgs/simple_rrt.png)

#### Circle RRT Path
A tree created from _Point A_ to _point B_ is shown below. The path found that navigates between the circle obstacles is colored in blue.

![circle_rrt.png](imgs/circle_rrt.png)

#### N RRT Path
A tree created from _Point A_ to _point B_ is shown below. The path found that navigates the "N" symbol is colored in blue.

![n_rrt.png](imgs/n_rrt.png)

## Implementation
Simple clone the package and run in an ipython shell ``%run rrt_challenges.py``.

All 3 challenges are run consecutively so in order to run the next challenge, close the figure that pops up after each challenge.
