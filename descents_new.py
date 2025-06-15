from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    """
    Класс для вычисления длины шага.

    Parameters
    ----------
    lambda_ : float, optional
        Начальная скорость обучения. По умолчанию 1e-3.
    s0 : float, optional
        Параметр для вычисления скорости обучения. По умолчанию 1.
    p : float, optional
        Степенной параметр для вычисления скорости обучения. По умолчанию 0.5.
    iteration : int, optional
        Текущая итерация. По умолчанию 0.

    Methods
    -------
    __call__()
        Вычисляет скорость обучения на текущей итерации.
    """
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Вычисляет скорость обучения по формуле lambda * (s0 / (s0 + t))^p.

        Returns
        -------
        float
            Скорость обучения на текущем шаге.
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    """
    Перечисление для выбора функции потерь.

    Attributes
    ----------
    MSE : auto
        Среднеквадратическая ошибка.
    MAE : auto
        Средняя абсолютная ошибка.
    LogCosh : auto
        Логарифм гиперболического косинуса от ошибки.
    Huber : auto
        Функция потерь Хьюбера.
    """
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    Базовый класс для всех методов градиентного спуска.

    Parameters
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float, optional
        Параметр скорости обучения. По умолчанию 1e-3.
    loss_function : LossFunction, optional
        Функция потерь, которая будет оптимизироваться. По умолчанию MSE.

    Attributes
    ----------
    w : np.ndarray
        Вектор весов модели.
    lr : LearningRate
        Скорость обучения.
    loss_function : LossFunction
        Функция потерь.

    Methods
    -------
    step(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Шаг градиентного спуска.
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов на основе градиента. Метод шаблон.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам. Метод шаблон.
    calc_loss(x: np.ndarray, y: np.ndarray) -> float
        Вычисление значения функции потерь.
    predict(x: np.ndarray) -> np.ndarray
        Вычисление прогнозов на основе признаков x.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация базового класса для градиентного спуска.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Функция потерь, которая будет оптимизирована.

        Attributes
        ----------
        w : np.ndarray
            Начальный вектор весов, инициализированный случайным образом.
        lr : LearningRate
            Экземпляр класса для вычисления скорости обучения.
        loss_function : LossFunction
            Выбранная функция потерь.
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Выполнение одного шага градиентного спуска.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами.
        """

        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для обновления весов. Должен быть переопределен в подклассах.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами. Этот метод должен быть реализован в подклассах.
        """
        step = -self.lr() * gradient.astype(np.float64)
        self.w = self.w.astype(np.float64)
        self.w += step
        return step

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для вычисления градиента функции потерь по весам. Должен быть переопределен в подклассах.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление значения функции потерь с использованием текущих весов.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        float
            Значение функции потерь.
        """
        y_pred = self.predict(x)
        if self.loss_function == LossFunction.MSE:
            return np.mean((y_pred - y) ** 2)
        elif self.loss_function == LossFunction.MAE:
            return np.mean(np.abs(y_pred - y))
        elif self.loss_function == LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(y_pred - y)))
        elif self.loss_function == LossFunction.Huber:
            error = y_pred - y
            delta = 1.0  # Huber delta
            return np.mean(np.where(np.abs(error) <= delta, 
                                0.5 * error**2, 
                                delta * (np.abs(error) - 0.5 * delta)))
        else:
            raise ValueError("Unknown loss function")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Расчет прогнозов на основе признаков x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        np.ndarray
            Прогнозируемые значения.
        """
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Класс полного градиентного спуска.
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов на основе градиента.
        """
        step = -self.lr() * gradient
        self.w += step
        return step

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по весам.
        """
        y_pred = self.predict(x)
        error = y_pred - y
        
        if self.loss_function == LossFunction.MSE:
            gradient = 2 * x.T @ error / len(y)
        elif self.loss_function == LossFunction.MAE:
            gradient = x.T @ np.sign(error) / len(y)
        elif self.loss_function == LossFunction.LogCosh:
            gradient = x.T @ np.tanh(error) / len(y)
        elif self.loss_function == LossFunction.Huber:
            delta = 1.0  # Huber delta
            error = y_pred - y
            condition = np.abs(error) <= delta
            gradient = (x.T @ (np.where(condition, error, delta * np.sign(error)))) / len(y)
        else:
            raise ValueError("Unknown loss function")
        
        return gradient


class StochasticDescent(VanillaGradientDescent):
    """
    Класс стохастического градиентного спуска с улучшенной обработкой батчей.

    Parameters
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float, optional
        Скорость обучения. По умолчанию 1e-3.
    batch_size : int, optional
        Размер мини-пакета. По умолчанию 50.
    loss_function : LossFunction, optional
        Функция потерь. По умолчанию MSE.
    replace : bool, optional
        Производить ли выборку с заменой. По умолчанию False.

    Attributes
    ----------
    batch_size : int
        Размер мини-пакета.
    replace : bool
        Флаг выборки с заменой.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE, replace: bool = False):
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = min(batch_size, dimension)  # Не больше размерности
        self.replace = replace

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по мини-пакету с проверками.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам.

        Raises
        ------
        ValueError
            Если размеры x и y не совпадают или данные пусты.
        """
        if len(x) != len(y):
            raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
        if len(y) == 0:
            raise ValueError("Cannot calculate gradient on empty data")
        
        # Адаптивный размер батча если выборка слишком мала
        current_batch_size = min(self.batch_size, len(y))
        
        # Выбор индексов для батча
        if self.replace:
            batch_indices = np.random.choice(len(y), size=current_batch_size, replace=True)
        else:
            batch_indices = np.random.choice(len(y), size=current_batch_size, replace=False)
        
        try:
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]
            return super().calc_gradient(x_batch, y_batch)
        except IndexError as e:
            raise IndexError(f"Batch indices out of bounds. Check your data dimensions. Error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error calculating gradient: {str(e)}")

    def set_batch_size(self, batch_size: int):
        """Безопасное обновление размера батча"""
        self.batch_size = max(1, min(batch_size, self.dimension))


class MomentumDescent(VanillaGradientDescent):
    """
    Класс градиентного спуска с моментом.

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    alpha : float
        Коэффициент момента.
    h : np.ndarray
        Вектор момента для весов.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием момента.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация класса градиентного спуска с моментом.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием момента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        # Обновляем вектор момента
        self.h = self.alpha * self.h + self.lr() * gradient
        
        # Обновляем веса
        step = -self.h
        self.w += step
        
        return step


class Adam(VanillaGradientDescent):
    """
    Класс градиентного спуска с адаптивной оценкой моментов (Adam).

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    eps : float
        Малая добавка для предотвращения деления на ноль.
    m : np.ndarray
        Векторы первого момента.
    v : np.ndarray
        Векторы второго момента.
    beta_1 : float
        Коэффициент распада для первого момента.
    beta_2 : float
        Коэффициент распада для второго момента.
    iteration : int
        Счетчик итераций.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием адаптивной оценки моментов.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация класса Adam.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием адаптивной оценки моментов.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.iteration += 1
        
        # Обновляем моменты
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        
        # Вычисляем моменты
        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)
        
        step = -self.lr() * m_hat / (np.sqrt(v_hat) + self.eps)
        self.w += step
        
        return step


class BaseDescentReg(BaseDescent):
    """
    Базовый класс для градиентного спуска с L2-регуляризацией.

    Параметры
    ----------
    *args : tuple
        Аргументы, передаваемые в базовый класс.
    mu : float, optional
        Коэффициент L2-регуляризации. По умолчанию 0.
    **kwargs : dict
        Ключевые аргументы, передаваемые в базовый класс.

    Атрибуты
    ----------
    mu : float
        Коэффициент L2-регуляризации.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь с учетом L2 регуляризации.
    calc_loss(x: np.ndarray, y: np.ndarray) -> float
        Вычисление полной функции потерь (основная + регуляризация).
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        Инициализация базового класса для градиентного спуска с регуляризацией.

        Parameters
        ----------
        mu : float
            Коэффициент L2-регуляризации (μ).
            Определяет силу штрафа за большие значения весов.
        """
        super().__init__(*args, **kwargs)
        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление полного градиента (основной + регуляризация).

        Формула:
        ∇Q(w) = ∇L(w) + μ * w
        где L(w) - основная функция потерь,
        μ - коэффициент регуляризации.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Матрица признаков.
        y : np.ndarray, shape (n_samples,)
            Вектор целевых значений.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Полный градиент функции потерь.
        """
        # Основной градиент от родительского класса
        main_gradient = super().calc_gradient(x, y)
        
        # Градиент регуляризационного члена (μ * w)
        reg_gradient = self.mu * self.w
        
        # Проверка совпадения размерностей
        if main_gradient.shape != reg_gradient.shape:
            raise ValueError(
                f"Несовпадение размеров: основной градиент {main_gradient.shape}, "
                f"градиент регуляризации {reg_gradient.shape}"
            )
        
        return main_gradient + reg_gradient

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление полной функции потерь: L(w) + μ/2 * ||w||².

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Матрица признаков.
        y : np.ndarray, shape (n_samples,)
            Вектор целевых значений.

        Returns
        -------
        float
            Значение полной функции потерь.
        """
        # Основная функция потерь
        base_loss = super().calc_loss(x, y)
        
        # L2-регуляризационный член (μ/2 * ||w||²)
        reg_loss = 0.5 * self.mu * np.sum(self.w ** 2)
        
        return base_loss + reg_loss

    def get_regularization_loss(self) -> float:
        """
        Вычисляет только значение регуляризационного члена.

        Returns
        -------
        float
            Значение регуляризации: μ/2 * ||w||²
        """
        return 0.5 * self.mu * np.sum(self.w ** 2)


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Класс полного градиентного спуска с регуляризацией.
    """

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь с учетом L2 регуляризации по весам.
        """
        gradient = super().calc_gradient(x, y)
        
        l2_gradient = 2 * self.mu * self.w
        return gradient + l2_gradient


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Класс стохастического градиентного спуска с регуляризацией.
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Класс градиентного спуска с моментом и регуляризацией.
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Класс адаптивного градиентного алгоритма с регуляризацией (AdamReg).
    """


def get_descent(descent_config: dict) -> BaseDescent:
    """
    Создает экземпляр класса градиентного спуска на основе предоставленной конфигурации.

    Параметры
    ----------
    descent_config : dict
        Словарь конфигурации для выбора и настройки класса градиентного спуска. Должен содержать ключи:
        - 'descent_name': строка, название метода спуска ('full', 'stochastic', 'momentum', 'adam').
        - 'regularized': булево значение, указывает на необходимость использования регуляризации.
        - 'kwargs': словарь дополнительных аргументов, передаваемых в конструктор класса спуска.

    Возвращает
    -------
    BaseDescent
        Экземпляр класса, реализующего выбранный метод градиентного спуска.

    Исключения
    ----------
    ValueError
        Вызывается, если указано неправильное имя метода спуска.

    Примеры
    --------
    >>> descent_config = {
    ...     'descent_name': 'full',
    ...     'regularized': True,
    ...     'kwargs': {'dimension': 10, 'lambda_': 0.01, 'mu': 0.1}
    ... }
    >>> descent = get_descent(descent_config)
    >>> isinstance(descent, BaseDescent)
    True
    """
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))