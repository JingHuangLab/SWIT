# load dependencies
import os
import re
import json
import tempfile
import sys

# --------- change these path variables as required
swit_dir= os.path.expanduser("~/tool/swit")
task_name = "task_test"
ckpt_path = os.path.join(swit_dir,"examples",task_name,"lightning_logs","version_5434400","checkpoints","epoch=49-step=799.ckpt")
gen_model_dir = os.path.join(swit_dir,"gen_models")
output_dir = os.path.join(swit_dir,"examples",task_name,"RL_practice","test1")
nstep = 10
n_mols= 800000  #When the set number of steps is not enough to generate the set number of molecules, this parameter has no effect.
low_mw = 200
high_mw = 650
upper_score=30


# --------- do not change
# generate a folder to store the results

try:
    mkdir_cmd = "mkdir -p "+output_dir
    os.system(mkdir_cmd)
except FileExistsError:
    pass

# Setting up the configuration
# initialize the dictionary
configuration = {
    "version": 2,                          # we are going to use REINVENT's old release
    "run_type": "reinforcement_learning"   # other run types: "sampling", "validation",
                                           #                  "transfer_learning",
                                           #                  "scoring" and "create_model"
                                           # didn't included in swit 
}
# add block to specify whether to run locally or not and
# where to store the results and logging
configuration["logging"] = {
    "sender": "http://127.0.0.1",          # only relevant if "recipient" is set to "remote"
    "recipient": "local",                  # either to local logging or use a remote REST-interface
    "logging_frequency": 1000,               # log every x-th steps
    "logging_path": os.path.join(output_dir,"progress.log"), # load this folder in tensorboard
    "resultdir": os.path.join(output_dir,"results"),         # will hold the compounds (SMILES) and summaries
    "job_name": "Reinforcement learning demo",                # set an arbitrary job name for identification
    "job_id": "demo"                       # only relevant if "recipient" is set to "remote"
}
# add the "parameters" block
configuration["parameters"] = {}

# add a "diversity_filter"
configuration["parameters"]["diversity_filter"] =  {
    "name": "IdenticalTopologicalScaffold",     # other options are: "IdenticalTopologicalScaffold", "IdenticalMurckoScaffold"
                                           #                    "NoFilter" and "ScaffoldSimilarity"
                                           # -> use "NoFilter" to disable this feature
    "nbmax": 10,                           # the bin size; penalization will start once this is exceeded
    "minscore": 0,                         # the minimum total score to be considered for binning
    "minsimilarity": 0.2                   # the minimum similarity to be placed into the same bin
}

# prepare the inception (we do not use it in this example, so "smiles" is an empty list)
configuration["parameters"]["inception"] = {
    "smiles": [],                          # fill in a list of SMILES here that can be used (or leave empty)
    "memory_size": 100,                    # sets how many molecules are to be remembered
    "sample_size": 10                      # how many are to be sampled each epoch from the memory
}

# set all "reinforcement learning"-specific run parameters
configuration["parameters"]["reinforcement_learning"] = {
    "prior": os.path.join(gen_model_dir, "data/augmented.prior"), # path to the pre-trained model
    "agent": os.path.join(gen_model_dir, "data/augmented.prior"), # path to the pre-trained model
    "n_steps": nstep,                      # the number of epochs (steps) to be performed; often 1000
    "n_mols" : n_mols,                     # the number of new molecules to be generated. Generation stops 
                                           # when the number of epochs or the number of new molecules reaches a set value. 
                                           # if it is set to None, that is, run nstep epochs.
    "sigma": 128,                          # used to calculate the "augmented likelihood", see publication
    "learning_rate": 0.0001,               # sets how strongly the agent is influenced by each epoch
    "batch_size": 128,                     # specifies how many molecules are generated per epoch
    "reset": 0,                            # if not '0', the reset the agent if threshold reached to get
                                           # more diverse solutions
    "reset_score_cutoff": 0.5,             # if resetting is enabled, this is the threshold
    "margin_threshold": 50                 # specify the (positive) margin between agent and prior
}

# Define the scoring function
# prepare the scoring function definition and add at the end
scoring_function = {
    "name": "custom_sum",                  # this is our default one (alternative: "custom_sum")
    "parallel": False,                     # sets whether components are to be executed
                                           # in parallel; note, that python uses "False" / "True"
                                           # but the JSON "false" / "true"

    # the "parameters" list holds the individual components
    "parameters": [
    {
        "component_type": "molecular_weight",
        "name": "Molecular weight",
        "weight": 1,
        "model_path": None,
        "smiles": [],
        "specific_parameters": {
            "transformation_type": "double_sigmoid",
            "high": high_mw,
            "low": low_mw,
            "coef_div": high_mw,
            "coef_si": 20,
            "coef_se": 20,
            "transformation": True,
        }
    },
    # add component: an activity model
    {
        "component_type": "cdock_score", # this is a scikit-learn model, returning
                                               # target-specific score values
        "name": "CDock Score",                 # arbitrary name for the component
        "weight": 2,                           # the weight ("importance") of the component (default: 1)
        "model_path": ckpt_path,                # absolute model path
        "smiles": [],                          # list of SMILES (not required for this component)
        "specific_parameters": {
            "transformation_type": "sigmoid",  # see description above
            "high": upper_score,                        # parameter for sigmoid transformation
            "low": 0,                          # parameter for sigmoid transformation
            "k": 0.2,                          # parameter for sigmoid transformation
            "scikit": "regression",            # model can be "regression" or "classification"
            "transformation": True,            # enable the transformation
        }
    },
]}
configuration["parameters"]["scoring_function"] = scoring_function

# write the configuration file to the disc
configuration_JSON_path = os.path.join(output_dir,"RL_config.json")
with open(configuration_JSON_path, 'w') as f:
    json.dump(configuration, f, indent=4, sort_keys=True)

