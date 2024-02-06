from environments.toy_env import ToyEnv
from learners.iterative_sieve_critic_lbfgs import IterativeSieveLearnerLBFGS
from utils.policy_evaluation import evaluate_policy
from policies.generic_policies import EpsilonSmoothPolicy
from policies.toy_env_policies import ThresholdPolicy
from utils.offline_dataset import OfflineRLDataset
from models.fnn_nuisance_model import FeedForwardNuisanceModel
from models.fnn_critic import FeedForwardCritic
from learners.iterative_sieve_critic import IterativeSieveLearner

def main():

    s_threshold = 2.0
    gamma = 0.9
    adversarial_lambda = 4.0
    batch_size = 1024
    num_sample = 10000
    worst_case = True

    device = None

    env = ToyEnv(s_init=s_threshold, adversarial=False)
    pi_e = ThresholdPolicy(env, s_threshold=s_threshold)
    pi_e_name = "pi_e"

    dataset_path_train = "tmp_dataset/train_data"
    dataset_path_test = "tmp_dataset/test_data"

    ## build datasets and save them

    pi_base = ThresholdPolicy(env, s_threshold=1.5)
    pi_b = EpsilonSmoothPolicy(env, pi_base=pi_base, epsilon=0.1)

    dataset = OfflineRLDataset()
    dataset.sample_new_trajectory(env=env, pi=pi_b, burn_in=1000,
                                num_sample=num_sample, thin=10)

    test_dataset = OfflineRLDataset()
    test_dataset.sample_new_trajectory(env=env, pi=pi_b, burn_in=1000,
                                    num_sample=num_sample, thin=10)

    dataset.apply_eval_policy(pi_e_name, pi_e)
    test_dataset.apply_eval_policy(pi_e_name, pi_e)

    dataset.save_dataset(dataset_path_train)
    test_dataset.save_dataset(dataset_path_test)

    dataset.to(device)
    test_dataset.to(device)


    # ## double check that dataset is loadable and looks correct

    # dataset_tmp = OfflineRLDataset.load_dataset(dataset_path_train)
    # dataset_tmp.to(device)
    # dl = dataset_tmp.get_batch_loader(batch_size=10)
    # for i, batch in enumerate(dl):
    #     for k, v in batch.items():
    #         print(k, v.shape)
    #         print(v)
    #     print("")
    #     if i > 10:
    #         break


    ## set up model

    s_dim = env.get_s_dim()
    num_a = env.get_num_a()

    # model_do = 0.05
    model_do = None
    model_config = {
        "s_embed_dim": 32,
        "s_embed_layers": [32],
        "s_embed_do": model_do,
        "a_embed_dim": 32,
        "sa_feature_dim": 64,
        "sa_feature_layers": [64],
        "sa_feature_do": model_do,
        "q_layers": [64, 64],
        "q_do": model_do,
        "beta_layers": [64, 64],
        "beta_do": model_do,
        "w_layers": [64, 64],
        "w_do": model_do,
        "eta_layers": [64, 64],
        "eta_do": model_do,
    }
    model = FeedForwardNuisanceModel(s_dim=s_dim, num_a=num_a, gamma=gamma,
                                     config=model_config, device=device)
    critic_class = FeedForwardCritic
    # critic_do = 0.05
    critic_do = None
    critic_config = {
        "s_embed_dim": 8,
        "s_embed_layers": [8],
        "s_embed_do": critic_do,
        "a_embed_dim": 8,
        "critic_layers": [16],
        "critic_do": critic_do,
    }
    critic_kwargs = {
        "s_dim": s_dim,
        "num_a": num_a,
        "config": critic_config
    }

    ## train mdoel on all moments

    learner = IterativeSieveLearner(
        nuisance_model=model, gamma=gamma,
        adversarial_lambda=adversarial_lambda,
        train_q_beta=True, train_eta=False, train_w=False,
        worst_case=worst_case, use_dual_cvar=True,
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

    learner.train(
        dataset, pi_e_name=pi_e_name, verbose=True, device=device,
        # init_basis_func=env.bias_basis_func, num_init_basis=1,
        model_eval_freq=5, critic_eval_freq=5,
        init_basis_func=env.flexible_basis_func,
        num_init_basis=env.get_num_init_basis_func(),
        # num_init_basis=env.get_num_init_basis_func(),
        evaluate_pv_kwargs=evaluate_pv_kwargs, critic_class=critic_class,
        s_init=s_init, critic_kwargs=critic_kwargs,
    )
    model.save_model("tmp_model")

    # ## train model on Eta / W moments second
    # model.freeze_embeddings()
    # learner_2 = IterativeSieveLearner(
    #     nuisance_model=model, gamma=gamma, 
    #     adversarial_lambda=adversarial_lambda, worst_case=worst_case
    #     train_q_beta=False, train_eta=True, train_w=True,
    # )
    
    # learner_2.train(
    #     dataset, pi_e_name=pi_e_name, verbose=True, device=device,
    #     init_basis_func=env.bias_basis_func, num_init_basis=1,
    #     model_eval_freq=5, critic_eval_freq=5,
    #     # init_basis_func=env.flexible_basis_func,
    #     # num_init_basis=env.get_num_init_basis_func(),
    #     # model_lr=1e-4,
    #     # num_init_basis=env.get_num_init_basis_func(),
    #     evaluate_pv_kwargs=evaluate_pv_kwargs, critic_class=critic_class,
    #     s_init=s_init, critic_kwargs=critic_kwargs,
    # )
    # model.save_model("tmp_model_2")

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
        worst_case=worst_case,
    )
    dr_pv_dual = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=True,
        worst_case=worst_case,
    )
    dr_pv_norm = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=False,
        normalize=True, worst_case=worst_case,
    )
    dr_pv_dual_norm = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=True,
        worst_case=worst_case,
    )
    print(f"EVALUATING FINAL BEST MODEL:")
    print(f"Q-estimated v(pi_e): {q_pv}")
    print(f"W-estimated v(pi_e): {w_pv}")
    print(f"W-estimated v(pi_e) (normalized): {w_pv_norm}")
    print(f"DS/DV-estimated v(pi_e): {dr_pv}")
    print(f"DS/DV-estimated v(pi_e) (dual): {dr_pv_dual}")
    print(f"DS/DV-estimated v(pi_e) (normalized): {dr_pv_norm}")
    print(f"DS/DV-estimated v(pi_e) (normalized, dual): {dr_pv_dual_norm}")
    print("")

    env_eval = ToyEnv(s_init=s_threshold, adversarial=True,
                        adversarial_lambda=adversarial_lambda)
    pi_e_val = evaluate_policy(env_eval, pi_e, gamma, min_prec=1e-4)
    print(f"true v(pi_e): {pi_e_val}")
    print("")


if __name__ == "__main__":
    main()