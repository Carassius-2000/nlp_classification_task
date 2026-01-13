import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud


def plot_hist_with_kde(data, x_label: str, font_size: int = 18):
    """
    Строит гистограмму распределения.

    Визуализирует распределение данных с помощью гистограммы,
    добавляет вертикальные линии для среднего и медианы, а также отображает
    стандартное отклонение в заголовке.

    Parameters
    ----------
    data : array-like
        Массив или подобная массиву структура данных для визуализации.
    x_label : str
        Подпись оси X (название визуализируемой метрики/признака).
    font_size : int, default=18
        Базовый размер шрифта для всех текстовых элементов графика.
    """
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
    """
    Создает и отображает облако слов на основе частотного словаря токенов.

    Функция генерирует визуализацию в виде облака слов, где размер каждого
    слова пропорционален его частоте в предоставленном словаре.

    Parameters
    ----------
    token_counter : dict[str, int]
        Словарь, где ключи - строки (слова/токены), значения - целые числа
        (частота встречаемости).
    plot_title : str
        Заголовок графика. Обычно описывает анализируемый набор данных или
        контекст использования облака слов.
    """
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
    """
    Создает горизонтальную столбчатую диаграмму для визуализации топ-N токенов.

    Функция строит график, отображающий наиболее часто встречающиеся
    токены и их частоты. Используется для анализа распределения частот слов или
    других текстовых единиц в корпусе.

    Parameters
    ----------
    token_counter : collections.Counter или аналогичный объект
        Объект, поддерживающий метод most_common(), возвращающий список пар
        (токен, частота). Обычно это Counter из модуля collections.
    plot_title : str
        Заголовок графика.
    y_label : str
        Подпись оси Y (обычно "Токены").
    max_token_count : int, default=30
        Максимальное количество токенов для отображения на графике
        (топ-N по частоте).
    font_size : int, default=18
        Базовый размер шрифта для всех текстовых элементов графика.
    """
    plt.figure(figsize=(19, 8))
    plt.title(plot_title, fontsize=font_size)
    sns.barplot(dict(token_counter.most_common(max_token_count)), n_boot=0, orient="h")
    plt.xlabel("Количество", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(axis="x")
    plt.show()
