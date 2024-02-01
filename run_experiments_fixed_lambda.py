from multiprocessing import Process, Queue
import json
import os
import pandas
from tqdm import tqdm
from itertools import product

from environments.toy_env import ToyEnv
from utils.policy_evaluation import evaluate_policy
from policies.generic_policies import EpsilonSmoothPolicy
from policies.toy_env_policies import ThresholdPolicy
from utils.offline_dataset import OfflineRLDataset
from models.fnn_nuisance_model import FeedForwardNuisanceModel
from models.fnn_critic import FeedForwardCritic
from learners.iterative_sieve_critic import IterativeSieveLearner


def main(config, results_path):
    num_rep = config["num_rep"]
    lambda_range = config["adversarial_lambda_values"]
    num_restart = config["num_restart"]
    job_queue = Queue()
    num_jobs = 0
    job_iter = product(range(num_rep), range(num_restart), lambda_range,
                       (True, False), (True, False))

    for rep_i, restart_i, lmbda, dual_cvar, sequential in job_iter:
        job = {"rep_i": rep_i, "adversarial_lambda": lmbda,
               "restart_i": restart_i, "dual_cvar": dual_cvar,
               "sequential": sequential}
        job_queue.put(job)
        num_jobs += 1

    procs = []
    results_queue = Queue()
    devices = config["devices"]
    for i in range(config["num_workers"]):
        device = devices[i % len(devices)]
        p_args = (job_queue, results_queue, config, device)
        p = Process(target=run_jobs_loop, args=p_args)
        procs.append(p)
        job_queue.put("STOP")
        p.start()

    all_results = []
    print("running experiments:")
    for _ in tqdm(range(num_jobs)):
        next_result = results_queue.get()
        all_results.extend(next_result)
        df = pandas.DataFrame(all_results)
        df.to_csv(results_path, index=False)
    for p in procs:
        p.join()

def run_jobs_loop(job_queue, results_queue, config, device):
    for job_kwargs in iter(job_queue.get, "STOP"):
        results = single_run(config=config, device=device, **job_kwargs)
        results_queue.put(results)

def single_run(config, rep_i, restart_i,
               adversarial_lambda, dual_cvar, sequential, device=None):


    env = ToyEnv(s_init=config["s_threshold"], adversarial=False)
    s_dim = env.get_s_dim()
    num_a = env.get_num_a()
    gamma = config["gamma"]
    s_threshold = config["s_threshold"]
    batch_size = config["batch_size"]

    pi_e = ThresholdPolicy(env, s_threshold=s_threshold)
    pi_e_name = config["pi_e_name"]

    model_config = config["model_config"]
    model = FeedForwardNuisanceModel(s_dim=s_dim, num_a=num_a, gamma=gamma,
                                     config=model_config, device=device)
    critic_class = FeedForwardCritic
    critic_config = config["critic_config"]
    critic_kwargs = {
        "s_dim": s_dim,
        "num_a": num_a,
        "config": critic_config
    }

    base_dataset_path_train = config["base_dataset_path_train"]
    base_dataset_path_test = config["base_dataset_path_test"]
    dataset_path_train = "_".join([base_dataset_path_train, str(rep_i+1)])
    dataset_path_test = "_".join([base_dataset_path_test, str(rep_i+1)])
    train_dataset = OfflineRLDataset.load_dataset(dataset_path_train)
    test_dataset = OfflineRLDataset.load_dataset(dataset_path_test)
    if device is not None:
        train_dataset.to(device)
        test_dataset.to(device)

    if sequential:
        learner = IterativeSieveLearner(
            nuisance_model=model, gamma=gamma, use_dual_cvar=dual_cvar,
            adversarial_lambda=adversarial_lambda,
            train_q_xi=True, train_eta=False, train_w=False,
        )
    else:
        learner = IterativeSieveLearner(
            nuisance_model=model, gamma=gamma, use_dual_cvar=dual_cvar,
            adversarial_lambda=adversarial_lambda,
            train_q_xi=True, train_eta=True, train_w=True,
        )

    s_init, a_init = env.get_s_a_init(pi_e)
    if device is not None:
        s_init = s_init.to(device)
        a_init = a_init.to(device)

    dl_test = test_dataset.get_batch_loader(batch_size=batch_size)
    evaluate_pv_kwargs = {
        "s_init": s_init, "a_init": a_init,
        "dl_test": dl_test, "pi_e_name": pi_e_name,
    }
    learner_kwargs = config["learner_kwargs"]
    learner.train(
        dataset=train_dataset, pi_e_name=pi_e_name, verbose=False,
        device=device, init_basis_func=env.bias_basis_func,
        num_init_basis=1, evaluate_pv_kwargs=evaluate_pv_kwargs,
        critic_class=critic_class, s_init=s_init,
        critic_kwargs=critic_kwargs, **learner_kwargs,
    )

    if sequential:
        #do a second training on eta / w moments
        learner_2 = IterativeSieveLearner(
            nuisance_model=model, gamma=gamma, use_dual_cvar=dual_cvar,
            adversarial_lambda=adversarial_lambda,
            train_q_xi=False, train_eta=True, train_w=True,
        )
        learner_2.train(
            dataset=train_dataset, pi_e_name=pi_e_name, verbose=False,
            device=device, init_basis_func=env.bias_basis_func,
            num_init_basis=1, evaluate_pv_kwargs=evaluate_pv_kwargs,
            critic_class=critic_class, s_init=s_init,
            critic_kwargs=critic_kwargs, **learner_kwargs,
        )

    model_path_base = config["base_model_path"]
    model_name = "model"
    model_name += f"_lambda={adversarial_lambda}"
    model_name += f"_sequantial={sequential}"
    model_name += f"_dual-cvar={dual_cvar}"
    model_name += f"_rep={rep_i}"
    model_path = os.path.join(model_path_base, model_name)
    model.save_model(model_path)

    ## evaluate model using 3 policy value estimators

    q_pv = model.estimate_policy_val_q(
        s_init=s_init, a_init=a_init, gamma=gamma
    )
    w_pv = model.estimate_policy_val_w(
        dl=dl_test, pi_e_name=pi_e_name,
    )
    w_pv_norm = model.estimate_policy_val_w(
        dl=dl_test, pi_e_name=pi_e_name, normalize=True,
    )
    dr_pv = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=False,
    )
    dr_pv_dual = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=True,
        hard_dual_threshold=False,
    )
    dr_pv_norm = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=False,
        normalize=True,
    )
    dr_pv_dual_norm = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=True,
        hard_dual_threshold=False, normalize=True,
    )
    pv_results = {
        "q": q_pv, "w": w_pv, "w_norm": w_pv_norm,
        "dr": dr_pv_dual, "dr_primal": dr_pv,
        "dr_norm": dr_pv_dual_norm, "dr_primal_norm": dr_pv_norm,
    }
    results = []
    for key, val in pv_results.items():
        row = {
            "rep_i": rep_i, "restart_i": restart_i, "dual_cvar": dual_cvar,
            "sequential": sequential, "lambda": adversarial_lambda,
            "est_policy_value": val, "estimator": key,
        }
        results.append(row)
    return results


if __name__ == "__main__":
    with open("experiment_config.json") as f:
        config = json.load(f)
    config["learner_kwargs"]["total_num_iterations"] = 0
    config["learner_kwargs"]["model_max_epoch_final"] = 2
    config["num_rep"] = 2
    config["adversarial_lambda_values"] = [1, 4]
    config["num_workers"] = 8
    main(config, "experiment_results.csv")

    # results = single_run(config=config, rep_i=0, adversarial_lambda=4,
    #                      dual_cvar=True, sequential=False, device=None)
    # for row in results:
    #     print(row)
    #     print()