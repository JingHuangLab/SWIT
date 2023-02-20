# SWIT
Source code, and data to reproduce results, for the manuscript

"A Simple Way to Incorporate Target Structural Information in Molecular Generative Models" by Wenyi Zhang, Kaiyue Zhang and Jing Huang.

## Requirements
* Install [Conda](https://conda.io/projects/conda/en/latest/index.html)
* Cuda-enabled GPU

## Installation
1. Clone this repository: 

        git clone https://github.com/JingHuangLab/SWIT.git
        
2. Go to the repository and create the Conda environment:
   
        conda env create -f environment.yml

3. Activate the environment:
   
        conda activate swit

## Usage
SWIT program independently trains the target-specific scoring model, and you can only perform the first step as needed. To reproduce the method shown in mentioned manuscript, please complete these two steps. The following commands use a subset of the AmpC target data as an example.
1. Train a target-specific scoring model. It outputs a score that can approximate the docking score of a given molecule for a specific target protein.
  
        python train_tss_model.py data/ampc_trainset_demo.csv task_test --testing_dataset_path data/ampc_testset_demo.csv --ncpu 6
  
    * `data/ampc_trainset_demo.csv` is the path of the training data. The first column of the data is the SMILES of the molecule, and the second column is the docking score, separated by commas and must contain column names.
    * `task_test` is the task name, you can set it to any name you like.
    * `--testing_dataset_path data/ampc_testset_demo.csv`  is the path of the testing data. (default = None) It uses the trained model to predict the test set you provide, and outputs the prediction and scatter plot in [`examples/task_test/preds/`](examples/task_test/preds/)
    * `--ncpu 6` is the number of cores available to this job. (default = 1)
    * `--epochs` is the number of epochs for model training. (default = 50)
  
2. The above model is embedded in the molecular generative model for guiding the generation of molecules that bind well to the target protein. 
   - Open `create_rl_config.py` and set parameters. For example, the file path, the same task name as the first step, and the number of iterations.
   - Generate json files.
  
            python create_rl_config.py
        
   - Run.
   
            python gen_models/input.py ./examples/task_test/RL_practice/test1/RL_config.json
        
     * `./examples/task_test/RL_practice/test1/RL_config.json` is the path to the json file just generated by your create_rl_config.py.
3. After running, you can go to the corresponding path to view the molecular generation results. For example, `./examples/task_test/RL_practice/test1/results/scaffold_memory.csv`
          
## Help
If you have any questions, please contact Kaiyue Zhang (zhangkaiyue@westlake.edu.cn)

## Authors
* Kaiyue Zhang (zhangkaiyue@westlake.edu.cn)
* Wenyi Zhang (zhangwenyi@westlake.edu.cn)
* Jing Huang (huangjing@westlake.edu.cn)

## Acknowledgments

Inspiration, code snippets, etc.
* [chemprop](https://github.com/chemprop/chemprop)
* [MolPAL](https://github.com/coleygroup/molpal)
* [REINVENT](https://github.com/MolecularAI/Reinvent)
