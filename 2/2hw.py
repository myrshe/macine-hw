# Выполните загрузку и предобработку файлов: выполнить нормализацию данных, избавиться от значений, которые коррелируют между собой (если таковые имеются)
# Построить 3-в график данных: x и y - все признаки без целевого значения (использовать уменьшение размерности пространства) z - целевое значение (SalePrice)
# Разбейте данные на x_train, y_train, x_test и y_test для оценки точности работы алгоритма.
# Посчитайте метрику RMSE. Если необходимо, попробуйте разные наборы параметров для получения лучшего результата.
# Постройте график зависимости ошибки от коэффициента регуляризации.
# Почитать про регуляризацию Lasso. Определить признак, который оказывает наибольшее влияние на целевое значение
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def main():
    dataset = pd.read_csv("AmesHousing.csv")
    # преобразуем категоричные признаки в числоввые
    encoder = LabelEncoder()
    for col in dataset.select_dtypes(exclude=np.number).columns:
        dataset[col] = encoder.fit_transform(dataset[col].astype(str))

    # удаляем сильно коррелируещие признаки
    correlation_matrix = dataset.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.8)]
    dataset = dataset.drop(columns=to_drop)

    features = dataset.drop(columns=["SalePrice"])
    target = dataset["SalePrice"]

    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        target,
        c=target,
        cmap='viridis',
        s=50
    )

    ax.set_xlabel("первая главная компонента")
    ax.set_ylabel("вторая главная компонента")
    ax.set_zlabel("цена продажи (SalePrice)")
    fig.colorbar(scatter, label="SalePrice")
    plt.title("трехмерное представление данных после PCA")
    plt.show()
    # обучающая и тестовая выборки
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    # подбираем параметр регуляризации
    reg_params = np.logspace(-2, 3, 100)
    rmse_values = []
    for param in reg_params:
        model = Lasso(alpha=param, max_iter=10000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmse_values.append(rmse)

    best_param = reg_params[np.argmin(rmse_values)]

    print(f'лучшее значение коэффициента регуляризации: {best_param:.3f}')

    plt.figure(figsize=(10, 6))
    plt.plot(reg_params, rmse_values, color='green', label='Lasso RMSE')
    plt.axvline(best_param, color='red', linestyle='--', label=f'Оптимальное α = {best_param:.3f}')
    plt.xscale('log')
    plt.xlabel('Параметр регуляризации (α)')
    plt.ylabel('RMSE')
    plt.title('Зависимость ошибки от силы регуляризации')
    plt.legend()
    plt.grid(True)
    plt.show()

    final_model = Lasso(alpha=best_param, max_iter=10000)
    final_model.fit(X_train, y_train)

    feature_names = dataset.drop(columns=["SalePrice"]).columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': final_model.coef_,
        'abs_coeff': np.abs(final_model.coef_)
    })

    importance_df = importance_df.sort_values(by='abs_coeff', ascending=False)

    top_feature = importance_df.iloc[0]
    print(f"\nНаиболее важный признак: {top_feature['feature']}")
    print(f"Его коэффициент: {top_feature['coefficient']:.2f}")

    plt.figure(figsize=(8, 6))
    sns.barplot(x='abs_coeff', y='feature', data=importance_df.head(5))
    plt.title('Топ-5 самых влиятельных признаков')
    plt.xlabel('Абсолютное значение коэффициента')
    plt.ylabel('Признак')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()