# pymultiprocessing-example

This is a very small example usage of Python's multiprocessing module to 
get familiar with the package for future use in a more complicated 
project.

Here, we have two processes running:
 - The webcam frame capture with a very simple face detector
 - The face location plotting process

The webcam frame capture process communicates the location of the faces 
in the frame to the second process, who's task is simply to plot the 
rectangles in order to illustrate the communication and its speed.

Since there are only two processes, this is a two-way communication, 
which is a perfect use-case for a Pipe. Plus, only one process sends 
information while the other one simply receives it to do something with 
it, which means the Pipe is unidirectional.
