import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metric_results(results, metric, ax, title):
    sns.boxplot(
        data=results.melt(
            id_vars="Centre",
            value_vars=[
                f"{metric}_LV_ED",
                f"{metric}_LV_ES",
                f"{metric}_MYO_ED",
                f"{metric}_MYO_ES",
                f"{metric}_RV_ED",
                f"{metric}_RV_ES",
            ],
            var_name="Region",
        ),
        hue="Centre",
        x="Region",
        y="value",
        ax=ax,
    )

    ax.legend(loc="lower left")
    ax.set_title(title)
    ax.grid(True)
    return ax


def save_results(model_name, show_results=False):
    results = pd.read_csv(f"models/{model_name}/results.csv", index_col=0)
    grouped_by_vendor = results.groupby(["Centre"]).mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1 = plot_metric_results(results, "Dice", ax1, "Dice Coefficient")
    ax2 = plot_metric_results(results, "IoU", ax2, "IoU")

    fig.suptitle(f"Results for {model_name}", fontsize=16)
    plt.subplots_adjust(wspace=0.3)

    plt.savefig(f"models/{model_name}/metrics.png")

    if show_results:
        plt.show()
    return fig


def main():
    return 0


if __name__ == "__main__":
    main()
