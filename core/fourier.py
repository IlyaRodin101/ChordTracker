"""Оконное преобразование Фурье (STFT) на основе ``numpy.fft``.

Математическая основа модуля — ряд Фурье. Периодический звук с основным тоном
``f0`` раскладывается в сумму синусоид:

.. math::

    x(t) = \\sum_{k=1}^{K} a_k \\sin(2\\pi k f_0 t + \\varphi_k),

где ``a_k`` — амплитуда ``k``-го обертона. Реальная песня не периодична, поэтому
ряд Фурье применяется к коротким окнам: для каждого окна вычисляются
коэффициенты Фурье, которые показывают, какие частоты присутствуют в звуке и
насколько они громкие. Такая последовательность спектров называется STFT
(short-time Fourier transform).

Модуль намеренно не использует ``librosa.stft``: преобразование написано
самостоятельно на ``numpy.fft.rfft``, чтобы было видно, как именно работает
ряд Фурье в программе.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

#: Частота дискретизации, к которой приводятся все треки (Гц).
DEFAULT_SAMPLE_RATE = 22050

#: Размер окна STFT в отсчётах (0,37 с при 22050 Гц).
DEFAULT_N_FFT = 8192

#: Шаг между соседними окнами в отсчётах.
DEFAULT_HOP_LENGTH = 2048

Number = Union[int, float]


def hann_window(n_fft: int, periodic: bool = True) -> np.ndarray:
    """Построить оконную функцию Хэннинга.

    Окно Хэннинга плавно гасит края анализируемого куска звука. Без него
    обрыв сигнала на границе окна добавляет в спектр «лишние» частоты
    (спектральное растекание), и ряд Фурье описывает не только реальный звук,
    но и сам факт обрезания. Формула периодического окна:

    .. math::

        w[n] = 0.5 - 0.5 \\cos\\left(\\frac{2\\pi n}{D}\\right),
        \\quad n = 0, \\dots, N-1,

    где ``D = N`` для периодического окна и ``D = N - 1`` для симметричного.

    Args:
        n_fft: Длина окна в отсчётах. Должна быть положительной.
        periodic: Если ``True``, строится периодическая версия окна (её
            использует STFT). Если ``False`` — симметричная версия, удобная
            для визуализации.

    Returns:
        Массив формы ``(n_fft,)`` с весами окна в диапазоне ``[0, 1]``.

    Raises:
        ValueError: Если ``n_fft`` не положительное число.
    """
    if n_fft <= 0:
        raise ValueError(f"n_fft должен быть положительным, получено {n_fft}")

    if n_fft == 1:
        return np.ones(1, dtype=np.float64)

    denominator = n_fft if periodic else n_fft - 1
    indices = np.arange(n_fft, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * indices / denominator)


@dataclass(frozen=True, eq=False)
class STFT:
    """Результат оконного преобразования Фурье.

    Поле :attr:`spectrum` содержит комплексные коэффициенты Фурье: модуль
    коэффициента — амплитуда частоты, аргумент — её фаза. Именно по модулю
    дальше строятся спектрограмма и хроматограмма.

    Attributes:
        spectrum: Комплексная матрица формы ``(n_fft // 2 + 1, n_frames)``.
            Строка — частотный бин, столбец — временное окно.
        sample_rate: Частота дискретизации исходного сигнала (Гц).
        n_fft: Размер окна STFT в отсчётах.
        hop_length: Шаг между окнами в отсчётах.
    """

    spectrum: np.ndarray
    sample_rate: int
    n_fft: int
    hop_length: int

    def __post_init__(self) -> None:
        """Проверить корректность размеров результата STFT.

        Raises:
            ValueError: Если спектр не двумерный, параметры неположительные
                или число частотных бинов не соответствует ``n_fft``.
        """
        if self.spectrum.ndim != 2:
            raise ValueError("spectrum должен быть двумерным массивом")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate должен быть положительным")
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft и hop_length должны быть положительными")
        if self.spectrum.shape[0] != self.n_fft // 2 + 1:
            raise ValueError(
                "Число строк spectrum должно равняться n_fft // 2 + 1: "
                f"{self.spectrum.shape[0]} != {self.n_fft // 2 + 1}"
            )

    @property
    def n_frames(self) -> int:
        """Количество временных окон в результате STFT."""
        return int(self.spectrum.shape[1])

    @property
    def magnitude(self) -> np.ndarray:
        """Модуль комплексного спектра — амплитуды частот.

        Returns:
            Вещественная матрица формы ``(n_fft // 2 + 1, n_frames)``.
        """
        return np.abs(self.spectrum)

    @property
    def power(self) -> np.ndarray:
        """Энергия (квадрат амплитуды) каждого частотного бина."""
        return np.abs(self.spectrum) ** 2

    @property
    def frequencies(self) -> np.ndarray:
        """Частоты строк спектра в герцах.

        Returns:
            Массив формы ``(n_fft // 2 + 1,)`` от 0 до частоты Найквиста.
        """
        return np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

    @property
    def times(self) -> np.ndarray:
        """Моменты времени начала окон в секундах.

        При ``center=True`` (используется по умолчанию в :func:`stft`) первый
        столбец относится к моменту ``0`` с.

        Returns:
            Массив формы ``(n_frames,)``.
        """
        return np.arange(self.n_frames, dtype=np.float64) * self.hop_length / self.sample_rate

    def to_db(
        self,
        ref: Optional[Number] = None,
        amin: float = 1e-10,
        top_db: Optional[float] = None,
    ) -> np.ndarray:
        """Перевести амплитудный спектр в децибелы для отображения.

        Args:
            ref: Опорная амплитуда. Если ``None``, берётся максимальная
                амплитуда спектра, и самый громкий бин получает 0 дБ.
            amin: Минимальная амплитуда, защищающая от ``log(0)``.
            top_db: Если задано, значения ограничиваются снизу уровнем
                ``max - top_db``.

        Returns:
            Матрица уровня звука в дБ той же формы, что и :attr:`spectrum`.

        Raises:
            ValueError: Если ``ref`` не положительный или ``amin`` <= 0.
        """
        if amin <= 0:
            raise ValueError("amin должен быть положительным")

        magnitude = self.magnitude
        if ref is None:
            ref_value = float(np.max(magnitude)) if magnitude.size else 0.0
            if ref_value <= 0.0:
                ref_value = 1.0
        else:
            ref_value = float(ref)
            if ref_value <= 0.0:
                raise ValueError("ref должен быть положительным")

        db = 20.0 * np.log10(np.maximum(amin, magnitude) / ref_value)

        if top_db is not None:
            if top_db < 0:
                raise ValueError("top_db должен быть неотрицательным")
            db = np.maximum(db, db.max() - top_db)

        return db


def stft(
    y: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    window: Optional[np.ndarray] = None,
    center: bool = True,
    pad_mode: str = "constant",
) -> STFT:
    """Вычислить оконное преобразование Фурье сигнала.

    Сигнал нарезается на перекрывающиеся окна длины ``n_fft`` с шагом
    ``hop_length``. Каждое окно умножается на функцию Хэннинга и раскладывается
    в ряд Фурье через ``numpy.fft.rfft``. Для вещественного сигнала достаточно
    половины спектра (от 0 до частоты Найквиста), поэтому используется
    ``rfft``.

    Args:
        y: Одномерный аудиосигнал.
        sr: Частота дискретизации сигнала в герцах.
        n_fft: Размер окна в отсчётах. При 22050 Гц окно 8192 даёт частотный
            шаг около 2,7 Гц.
        hop_length: Шаг между окнами в отсчётах.
        window: Собственное окно длины ``n_fft``. Если ``None``, применяется
            :func:`hann_window`.
        center: Если ``True``, сигнал дополняется по краям на ``n_fft // 2``
            отсчётов, чтобы первый и последний кадры были центрированы на
            реальных отсчётах. Это поведение совпадает с ``librosa.stft``.
        pad_mode: Режим дополнения ``numpy.pad`` при ``center=True``.
            По умолчанию ``"constant"`` — так же, как в ``librosa.stft``.
            Режим ``"reflect"`` уменьшает краевые артефакты.

    Returns:
        Объект :class:`STFT` с комплексным спектром и координатными осями.

    Raises:
        ValueError: Если сигнал не одномерный, параметры неположительные или
            длина пользовательского окна не равна ``n_fft``.
    """
    samples = np.asarray(y)
    if samples.ndim != 1:
        raise ValueError(f"Ожидался одномерный сигнал, получена форма {samples.shape}")
    if sr <= 0:
        raise ValueError("sr должен быть положительным")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft и hop_length должны быть положительными")

    if samples.dtype != np.float32:
        # float64, float16 и целые числа приводим к float32: этого достаточно
        # для аудио, а numpy.fft не поддерживает float16.
        samples = samples.astype(np.float32)

    if window is None:
        window_values = hann_window(n_fft).astype(samples.dtype, copy=False)
    else:
        window_values = np.asarray(window, dtype=samples.dtype)
        if window_values.ndim != 1 or window_values.size != n_fft:
            raise ValueError(
                f"window должен иметь длину n_fft={n_fft}, получено {window_values.shape}"
            )

    if center:
        pad_width = n_fft // 2
        try:
            padded = np.pad(samples, pad_width, mode=pad_mode)
        except ValueError:
            # Например, reflect требует, чтобы сигнал был длиннее отступа.
            padded = np.pad(samples, pad_width, mode="constant")
    else:
        padded = samples

    if padded.size < n_fft:
        padded = np.pad(padded, (0, n_fft - padded.size), mode="constant")

    # Представление "скользящее окно" не копирует сигнал: копия появляется
    # только при умножении на оконную функцию.
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    windowed_frames = frames * window_values
    spectrum = np.fft.rfft(windowed_frames, axis=1).T
    # numpy.fft всегда возвращает complex128; для аудио достаточно complex64,
    # как в librosa. Это вдвое экономит память.
    spectrum = spectrum.astype(np.complex64)

    return STFT(
        spectrum=spectrum,
        sample_rate=int(sr),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
    )


def hz_to_midi(frequencies: np.ndarray, tuning: float = 0.0) -> np.ndarray:
    """Перевести частоты в номера MIDI-нот.

    Используется равномерно темперированный строй:

    .. math::

        \\mathrm{midi} = 69 + 12 \\log_2\\left(\\frac{f}{440}\\right) - t,

    где ``t`` — поправка строя в полутонах. Если трек записан чуть выше или
    ниже 440 Гц, ``t`` сдвигает все ноты к ближайшим стандартным.

    Args:
        frequencies: Частоты в герцах (скаляр или массив).
        tuning: Поправка строя в полутонах, например ``+0.3``.

    Returns:
        Номера MIDI-нот той же формы, что и ``frequencies`` (вещественные).

    Raises:
        ValueError: Если среди частот есть отрицательные значения.
    """
    values = np.asarray(frequencies, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("Частоты не могут быть отрицательными")

    safe_values = np.maximum(values, np.finfo(np.float64).tiny)
    return 69.0 + 12.0 * np.log2(safe_values / 440.0) - float(tuning)


def frames_to_time(
    n_frames: int,
    sr: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Перевести индексы кадров STFT в секунды.

    Args:
        n_frames: Количество кадров.
        sr: Частота дискретизации в герцах.
        hop_length: Шаг между кадрами в отсчётах.

    Returns:
        Массив времени формы ``(n_frames,)``.

    Raises:
        ValueError: Если ``n_frames`` отрицательное или параметры неположительные.
    """
    if n_frames < 0:
        raise ValueError("n_frames не может быть отрицательным")
    if sr <= 0 or hop_length <= 0:
        raise ValueError("sr и hop_length должны быть положительными")

    return np.arange(n_frames, dtype=np.float64) * hop_length / sr
