import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_PATHS = [
    "experiment_results.csv",
]

TRUE_POLICY_VALUE_PATH = "results_true_policy_value.csv"

HUE_NAMES = {
    "q": "Q",
    "w": "W",
    "dr": "Orth"
}
HUE_ORDER = [HUE_NAMES["q"], HUE_NAMES["w"], HUE_NAMES["dr"]]

def main():
    results_df_list = []
    for path in RESULTS_PATHS:
        df = pd.read_csv(path, index_col=False)
        results_df_list.append(df)
    df_results = pd.concat(results_df_list, ignore_index=True)
    print(df_results.head())
    df_true_pv = pd.read_csv(TRUE_POLICY_VALUE_PATH, index_col=False)
    # df_true_pv = df_true_pv.set_index("lambda")
    # print(df_true_pv.head())
    # df = df_results.join(other=df_true_pv, on="lambda", how="outer")
    df = df_results[df_results["estimator"].isin(("q", "w", "dr"))]
    df["estimator"].replace(HUE_NAMES, inplace=True)

    df_robust = df.groupby(["rep_i", "lambda", "estimator"]).median().reset_index()
    # df_robust = df
    plt.figure(figsize=(8, 4))
    ax = sns.boxplot(data=df_robust, x="lambda", y="est_policy_value",
                hue="estimator", gap=0.2, hue_order=HUE_ORDER)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    [ax.axvline(x, color = 'k', linestyle='--') for x in [0.5, 1.5, 2.5, 3.5]] 
    # sns.pointplot(data=df_robust, x="lambda", y="true_policy_value",
    #               linestyles="none", color="red", markers="*")
    vals = df_true_pv["true_policy_value"]
    for i, v in enumerate(vals):
        plt.hlines(v, xmin=i-0.5, xmax=i+0.5, color='r', linestyles='--')
    plt.xlim(-0.5, len(vals)-0.5)
    plt.xlabel("Lambda", fontsize=12)
    plt.ylabel("Estimated Policy Value", fontsize=12)
    plt.tight_layout()
    plt.savefig("tmp_boxplot.pdf")


if __name__ == "__main__":
    main()