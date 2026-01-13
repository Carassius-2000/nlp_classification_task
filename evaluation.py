import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    matthews_corrcoef,
)


def evaluate_classification(y_true, y_pred) -> None:
    """Выводит основные метрики качества для задачи классификации.

    Отображает:
    - precision, recall, f1-score для каждого класса;
    - macro avg, weighted avg по этим метрикам;
    - accuracy;
    - Matthews Correlation Coefficient (MCC).

    Parameters
    ----------
    y_true : np.array
        Истинные значения целевой переменной.
    y_pred : np.array
        Предсказанные значения целевой переменной.
    """
    print(
        classification_report(
            y_true,
            y_pred,
            digits=3,
        )
    )
    print(f"MCC равен {matthews_corrcoef(y_true, y_pred):.3f}")


def plot_confusion_matrix(model, X, y_true, title: str, font_size: int = 14) -> None:
    """Визуализирует матрицу ошибок для задачи классификации.

    Строит нормализованную матрицу ошибок, где каждая ячейка показывает
    долю объектов класса относительно всех объектов этого класса.

    Parameters
    ----------
    model : estimator object
        Обученная модель scikit-learn-like.
    X : array-like
        Матрица признаков для оценки модели.
    y_true : np.array
        Истинные значения целевой переменной.
    title : str
        Заголовок графика.
    font_size : int
        Размер шрифта для заголовков и подписей осей, по умолчанию 14.
    """
    plt.rcParams.update({"font.size": font_size})
    matrix = ConfusionMatrixDisplay.from_estimator(
        model,
        X,
        y_true,
        values_format=".3g",
        normalize="true",
    )
    fig = matrix.ax_.get_figure()
    matrix.ax_.set_ylabel("Фактический класс")
    matrix.ax_.set_xlabel("Предсказанный класс")
    matrix.ax_.set_title(title)
    fig.set_figwidth(12)
    fig.set_figheight(7)
