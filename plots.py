import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud


def plot_hist_with_kde(data, x_label: str, font_size: int = 18):
    plt.figure(figsize=(19, 8))
    plt.title(
        rf"Гистрограмма распределения, $\sigma$ = {data.std():.1f}",
        fontsize=font_size,
    )
    ax = sns.histplot(data, kde=True, bins=100)
    ax.lines[0].set_color("red")

    plt.axvline(
        x=data.mean(),
        linewidth=6,
        linestyle="--",
        color="g",
        label=f"Среднее = {data.mean():.1f}",
    )
    plt.axvline(
        x=np.median(data),
        linewidth=6,
        linestyle="-.",
        color="m",
        label=f"Медиана = {np.median(data):.1f}",
    )
    plt.xlabel(x_label, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylabel("Количество", fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=font_size, shadow=True)
    plt.show()


def plot_wordcloud(token_counter: dict[str, int], plot_title: str) -> None:
    wordcloud = WordCloud(
        width=1_900,
        height=800,
        background_color="white",
        max_words=100,
        max_font_size=150,
        min_font_size=12,
        font_step=1,
        prefer_horizontal=0.9,
        relative_scaling=0.5,
        random_state=1,
        contour_width=1,
        contour_color="steelblue",
    ).generate_from_frequencies(token_counter)
    plt.figure(figsize=(19, 10))
    plt.title(
        plot_title,
        fontsize=18,
    )
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.show()


def plot_bar(
    token_counter,
    plot_title: str,
    y_label: str,
    max_token_count: int = 30,
    font_size: int = 18,
):
    plt.figure(figsize=(19, 8))
    plt.title(plot_title, fontsize=font_size)
    sns.barplot(dict(token_counter.most_common(max_token_count)), n_boot=0, orient="h")
    plt.xlabel("Количество", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(axis="x")
    plt.show()
