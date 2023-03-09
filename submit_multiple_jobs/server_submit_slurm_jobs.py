# Install cmdstanpy: https://mc-stan.org/cmdstanpy/installation.html

## This gives an example for how to run multiple slurm jobs on the server, where
# each job runs a different stan model with different settings.
# This script is the one that will be run from the command line. 
# This script then creates multiple slurm.sub files (using the template), each with
# a different parameter set.
# It then submits each of these slurm.sub files to the server.

import time
import argparse
import numpy as np
import os

import settings

slurm_template = 'sub_template.txt'
slurmtxt = open(slurm_template, "r+").read()


def create_job_from_setting(model_name, settings_name, iter_warmup, iter_sampling, max_incidents, covariates_cont_name, model_folder, savelabel, tract11_or_block12, partition = 'default_partition'):
    substr = slurmtxt.format(jobname='tree_{}_{}'.format(model_name, settings_name),
                            modelname=model_name, settings_name=settings_name, iter_warmup=iter_warmup, iter_sampling=iter_sampling, max_incidents=max_incidents, covariates_cont_name=covariates_cont_name, model_folder=model_folder, savelabel=savelabel, tract11_or_block12 = tract11_or_block12, partition=partition
                            )
    print(settings_name, model_name)
    jobfilename = '{}_{}.sub'.format(model_name, savelabel)
    with open(jobfilename, "w") as f:
        f.write(substr)
    os.system('sbatch --requeue {}'.format(jobfilename))
    os.system('mv {} oldsubfilesautogen/'.format(jobfilename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # some arguments should be able to come from command line
    # these are arguments specific to the model we want to fit
    parser.add_argument("savelabel", type=str)
    parser.add_argument("--iter_warmup", type=int, default=300)
    parser.add_argument("--iter_sampling", type=int, default=300)
    parser.add_argument("--max_incidents", type=int, default=-1)
    parser.add_argument("--model_folder", type=str,
                        default='stan_modeling/models')
    parser.add_argument("--covariates_cont_name", type=str,
                        default='covdef')
    parser.add_argument("--settings_name_list", type=str,
                        default='settings_name_list_default')
    parser.add_argument("--rootdirectory", type=str,
                        default='/home/username/projectname/')
    parser.add_argument("--tract11_or_block12", type=int, default=11)
    parser.add_argument('--partition', type=str, default='default_partition')


    # parse the arguments
    args = parser.parse_args()

    # just run all models in the model_folder
    model_names = [x[0:-5] for x in os.listdir(
        '{}/{}'.format(args.rootdirectory, args.model_folder)) if x[-5:] == '.stan']
    print('model names: ', model_names)

    settings_names_list_to_run = settings.settings_lists_dict[args.settings_name_list]
    print('settings: ', settings_names_list_to_run)

    for settings_name in settings_names_list_to_run:
        paramhash = '{}{}{}{}'.format(
            args.iter_warmup, args.iter_sampling, args.max_incidents, args.covariates_cont_name)
        savelabel = '{}_{}_{}'.format(args.savelabel, settings_name, paramhash)

        for model_name in model_names:
            create_job_from_setting(model_name, settings_name, iter_warmup = args.iter_warmup, iter_sampling = args.iter_sampling, max_incidents = args.max_incidents, covariates_cont_name = args.covariates_cont_name, model_folder = args.model_folder, savelabel=savelabel, tract11_or_block12 = args.tract11_or_block12, partition = args.partition)

        # to allow models to compile safely without race conditions
        time.sleep(60)
