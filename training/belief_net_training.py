


import multiprocessing as mp

from sys import 



# idea: spin up X games, sampling from various opponets

# each instance runs until it reaches a point where it needs model output, then stops

# once we have collected enough instances to from a large model batch, 
# run the batch and return the results to each worker

# continue until N samples are collected


num_workers = 100

with mp.Pool(processes=mp.cpu_count()) as pool:
    
