import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import warnings
#warnings.filterwarnings('ignore')

from multiprocessing import Pool
import os
import time

"""
In this code, we use the multi-processing library to run simulations in parallel and then save them to a file in serial fashion. The code is structured as follows:
    1. Slurm file that submits the job to the cluster, calls python script (this file) from command line
    
    2. The python script parses arguments in the main function, and then calls a "pipeline" function, which is where all the action is. 
    
    3. The pipeline function uses "imap_unordered" from the multiprocessing package to run things in parallel. See (https://superfastpython.com/multiprocessing-pool-imap_unordered/) for a tutorial. The "imap_unordered" function takes 2 arguments: a function and a generator. The function is the function that you want to run in parallel. The generator is a function that returns a sequence of values where each value is an argument to the function.
    
        A generator is a function that returns a sequence of values, one at a time. Think of it as a function that returns a list, but instead of loading the whole list into memory, it returns one value at a time. For example, the range function is a generator. Thus, range(1e100) returns a sequence of numbers from 0 to 1e100, without actually storing all of those numbers in memory. i.e., the range is defined similar to the following:

                    def range(n):
                        i = 0
                        while i < n:
                            yield i
                        
        
    4. The imap_unordered function thus loops through the generator and calls the function with each value. These function calls are what are done in the separate threads (in parallel). The functions are allowed to return a result, which are then themselves returned by imap_unordered as another generator.

    5. We then have a for loop that loops through the results imap_unordered is returning. The code inside the for loop is happening in sequence, so we can save the results to a file in serial fashion. 
    
"""

#######################################################

# generator function that returns a sequence of values that are the parameters to the simulator function
def parameter_generator(hyperparameters_to_parameter_generation, num_times_to_repeat_each_parameter_set = 10, number_parameters_to_generate = 1000):
    
    for i in range(number_parameters_to_generate):
        param1 = np.random.uniform(hyperparameters_to_parameter_generation['min'], hyperparameters_to_parameter_generation['max'])
        
        param2 = np.random.normal(hyperparameters_to_parameter_generation['mean'], hyperparameters_to_parameter_generation['sd'])
        
        for j in range(num_times_to_repeat_each_parameter_set):
            yield (param1, param2)

# Function that we want to run in parallel, for different parameter values
def simulator(tup):
    rand1, rand2 = tup

    product = rand1 * rand2 + np.random.rand() # do some computation

    return pd.DataFrame({'rand1': [rand1], 'rand2': [rand2], 'product': [product]})


import time

# do the same thing but in parallel using multiprocessing package
def pipeline(hyperparameters_to_parameter_generation, num_times_to_repeat_each_parameter_set, number_parameters_to_generate
                                      , n_nodes = None
                                      , save_file = 'results.csv'
                                      ):
    pool = Pool(n_nodes)
    
    # if the results file already exists, load it so we can append to it
    if os.path.exists(save_file):
        output_df = pd.read_csv(save_file)
    else:
        output_df = pd.DataFrame()
        
    newdone = 0
    
    generator = parameter_generator(hyperparameters_to_parameter_generation, num_times_to_repeat_each_parameter_set, number_parameters_to_generate)
    
    ts = time.time()
    
    minutes_freq_to_save = 5 #save the results file every 5 minutes so that we don't lose everything if the job crashes
    
    
    # Each simulator(argument) call runs in a separate thread, and returns it's result within the pool command. The pool imap_unordered then returns the results one by one (it is actually returning as a generator as well...)

    # In other words the following line is equivalent to the following, except that first for loop is running in parallel on different threads, and we don't have to wait for all the runs to finish before we start saving the results to a file:
        #  results = []
        #  for parameterset in generator:
            #  results.append(simulator(parameterset))
        # for result in results:
            # ...
    for result in pool.imap_unordered(simulator, generator):
        
        # This for loop is running in serial and so saving the file is thread safe. 
        
        #save result to file
        output_df = pd.concat([output_df,result],ignore_index=True)
        newdone += 1
        
        curtime = time.time()
        
        # been more than 5 minute(s) since last save
        if (curtime - ts) / 60 > minutes_freq_to_save:
            output_df.to_csv(save_file, index = False)
            newdone = 0
            ts = curtime

    output_df.to_csv(save_file, index = False)
    
    pool.close()
    return output_df



import argparse
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # add arguments to the parser
    parser.add_argument("runlabel", type = str)
    parser.add_argument("--hyperparameters_file", type = str, default = 'hyperparameters.csv')
    parser.add_argument("--num_times_to_repeat_each_parameter_set", type = int, default = 100)
    parser.add_argument("--number_parameters_to_generate", type = int, default = 1000)
    
    parser.add_argument("--results_folder", type = str, default = 'results/')
    
    args = parser.parse_args()
    
    save_file = args.results_folder + args.runlabel +  '_results.csv'
    
    hyperparameters = pd.read_csv(args.hyperparameters_file).to_dict('records')[0] #have a hyperparameters csv where the columsn are the hyperparameters and the rows are the values
    
    pipeline(
        hyperparameters
        , args.num_times_to_repeat_each_parameter_set
        , args.number_parameters_to_generate
    )
