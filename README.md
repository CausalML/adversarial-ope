# Efficient and Sharp Off-policy Evaluation in Robust Markov Decision Processes

Authors: Andrew Bennett, Nathan Kallus, Miruna Oprescu, Wen Sun, Kaiwen Wang


## Abstract

We study the evaluation of a policy under best- and worst-case perturbations to a Markov decision process (MDP), using transition observations from the original MDP, whether they are generated under the same or a different policy. This is an important problem when there is the possibility of a shift between historical and future environments, e.g. due to unmeasured confounding, distributional shift, or an adversarial environment. We propose a perturbation model that allows changes in the transition kernel densities up to a given multiplicative factor or its reciprocal, extending the classic marginal sensitivity model (MSM) for single time-step decision-making to infinite-horizon RL. We characterize the sharp bounds on policy value under this model -- i.e., the tightest possible bounds based on transition observations from the original MDP -- and we study the estimation of these bounds from such transition observations. We develop an estimator with several important guarantees: it is semiparametrically efficient, and remains so even when certain necessary nuisance functions, such as worst-case Q-functions, are estimated at slow, nonparametric rates. Our estimator is also asymptotically normal, enabling straightforward statistical inference using Wald confidence intervals. Moreover, when certain nuisances are estimated inconsistently, the estimator still provides valid, albeit possibly not sharp, bounds on the policy value. We validate these properties in numerical simulations. The combination of accounting for environment shifts from train to test (robustness), being insensitive to nuisance-function estimation (orthogonality), and addressing the challenge of learning from finite samples (inference) together leads to credible and reliable policy evaluation.

## Publications

- Currently available in prepprint form [here](https://arxiv.org/abs/2404.00099)

## Data

The main experiments run only on synthetic data, which is generated via the `build_datasets.py` script. See execution details below.

The additional sepsis investigation uses a blackbox sepsis management simulator that was trained using electronic health record data from the MIMIC-III dataset (Johnson et al. (2016)). See setup instructions below for details on how to obtain, and our paper for biliographical details.

## Code

### Layout

The following is a quick summary of the code structure:
1. *environments* contains the Environments 
2. *policies* contains different policies we experimented with
3. *models* contains nn.Module code for modelling the Q / beta / w functions (along with the critic functions for minimax methods)
4. *learners* contains the learning algorithms (both robust FQI for Q estimation, and sieve-based minimax learning for W estimation)
5. *utils* contains some useful generic utilities

### Setup

Versions of important libraries used are in requirements.txt, which can be installed via `pip install -r requirements.txt`. Note that tensorflow and keras are only needed to run the additional sepsis investigation experiments. All experiments were run in python 3.10.

In order to run the sepsis investigation experiments, the `gym_sepsis` directory must also be populated with the contents of the `gym_sepsis` folder 
from [the gym-sepsis repository](https://github.com/akiani/gym-sepsis).


### Execution

In order to run the main synthetic experiments and repreoduce experimental results, you need to:
1. run `build_datasets.py` to build the datasets for the N dataset replications
2. run `run_experiments.py` to run the main experiments. This by default uses the config file at `configs/experiment_config.json`, but can be changed using --config command line arg
3. run `get_true_policy_values.py` to compute the true worst-case policy values for each Lambda (needed for creating plots)
4. run `create_results_plots.py` to build the result plot and table of MSE values

In order to run the additional sepsis management investigation in the appendix and reproduce results, you need to:
1. run `sepsis_investigation_build_offline_data.ipynb` in order to train the behavioral/target policies and build the offline dataset
2. run `sepsis_investigation_run_estimators.ipynb` in order to run the estimators and compute the results
3. run `get_true_sepsis_policy_values.py` to compute the true policy value of target policy (for Lambda=1)

Note that some parts of the configuration file might need to be modified for code to run correctly on the target machine. Of important note:
- "num_workers" specifies the number of parallel processes to run for experiments (default is 4, but this can be increased to speed up experiments)
- "devices" lists the devices used in torch (i.e. GPUs and/or CPUs) for running experiments on. Each worker process will be asigned a device from the list of available devices procided in the config file in a round-robin fashion. By default we set devices=[0], meaning that everything can be run on device 0, but this can be modified to run on multiple GPU, or this key can be deleted from the config file to just run everything on CPU.

In addition, the experiment can be run in different ways, e.g. with more replications or different hyperparameters, by modifying the config appropriately.


## License

All source files in this repository other than that within the `gym_sepsis`
directory, unless explicitly mentioned otherwise, are released under the
Apache 2.0 license, the text of which can be found in the LICENSE file.
