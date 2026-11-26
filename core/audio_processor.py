"""Загрузка аудио, HPSS и хроматограммы на основе ряда Фурье.

Модуль связывает математику из :mod:`core.fourier` с музыкальными понятиями:

* оконное преобразование Фурье раскладывает звук на частоты;
* HPSS (harmonic/percussive source separation) делит спектр на гармоническую
  часть (гитара, клавиши, бас) и ударную (барабаны, тарелки);
* частоты переводятся в 12 классов высоты тона (C, C#, ..., B), и получается
  хроматограмма — основа распознавания аккордов;
* отдельная басовая хроматограмма строится по низким частотам (65–260 Гц),
  чтобы потом определять корень аккорда.

Ряд Фурье проявляется здесь дважды. Во-первых, каждый кадр STFT — это набор
коэффициентов Фурье. Во-вторых, шаблоны аккордов (задача 3) повторяют
структуру обертонов ``f0, 2f0, 3f0, ...`` из ряда Фурье.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import librosa
import numpy as np

from core.fourier import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_SAMPLE_RATE,
    STFT,
    hz_to_midi,
    stft,
)

#: Частота ноты C2 (MIDI 36) — нижняя граница анализируемого диапазона.
C2_HZ = 65.40639

#: Частота ноты C4 (MIDI 60) — верхняя граница басовой хроматограммы.
C4_HZ = 261.62557

#: Верхняя граница обычной хроматограммы: выше 2 кГц преобладают обертоны и шум.
DEFAULT_FMAX_HZ = 2000.0

#: Названия 12 классов высоты тона, начиная с C.
PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

PathLike = Union[str, Path]


@dataclass(frozen=True, eq=False)
class AudioAnalysis:
    """Результат анализа трека, необходимый для распознавания аккордов.

    Attributes:
        path: Путь к исходному аудиофайлу.
        sample_rate: Частота дискретизации после приведения к 22050 Гц.
        duration: Длительность трека в секундах.
        tuning: Оценка отклонения строя от 440 Гц в полутонах.
        spectrogram_db: Спектрограмма полного сигнала в децибелах, форма
            ``(n_fft // 2 + 1, n_frames)``.
        harmonic_magnitude: Гармоническая часть спектрограммы (амплитуды).
        percussive_magnitude: Ударная часть спектрограммы (амплитуды).
        chroma: Хроматограмма гармонической части, форма ``(12, n_frames)``.
        bass_chroma: Хроматограмма низких частот (65–260 Гц), форма
            ``(12, n_frames)``.
        frequencies: Частоты строк спектрограммы в герцах.
        times: Время столбцов в секундах.
        n_fft: Размер окна STFT.
        hop_length: Шаг между окнами STFT.
    """

    path: str
    sample_rate: int
    duration: float
    tuning: float
    spectrogram_db: np.ndarray
    harmonic_magnitude: np.ndarray
    percussive_magnitude: np.ndarray
    chroma: np.ndarray
    bass_chroma: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    n_fft: int
    hop_length: int

    @property
    def n_frames(self) -> int:
        """Количество временных кадров в хроматограмме."""
        return int(self.chroma.shape[1])

    def mean_chroma(self) -> np.ndarray:
        """Усреднить хроматограмму по всему треку.

        Returns:
            Вектор длины 12 с относительной силой каждой ноты.
        """
        if self.n_frames == 0:
            return np.zeros(12, dtype=np.float32)
        return np.mean(self.chroma, axis=1)

    def mean_bass_chroma(self) -> np.ndarray:
        """Усреднить басовую хроматограмму по всему треку.

        Returns:
            Вектор длины 12 с относительной силой каждой низкой ноты.
        """
        if self.n_frames == 0:
            return np.zeros(12, dtype=np.float32)
        return np.mean(self.bass_chroma, axis=1)

    def strongest_pitch_classes(self, count: int = 3) -> list[str]:
        """Назвать самые сильные ноты трека по средней хроматограмме.

        Args:
            count: Сколько нот вернуть (от 1 до 12).

        Returns:
            Список названий нот, отсортированный по убыванию силы.

        Raises:
            ValueError: Если ``count`` вне диапазона 1–12.
        """
        if not 1 <= count <= 12:
            raise ValueError("count должен быть от 1 до 12")

        mean = self.mean_chroma()
        order = np.argsort(mean)[::-1][:count]
        return [PITCH_CLASS_NAMES[int(index)] for index in order]


def load_audio(
    path: PathLike,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Загрузить аудиофайл, привести его к моно и нужной частоте дискретизации.

    Для ``.mp3`` внутри ``librosa`` используется ``ffmpeg``, для ``.wav`` —
    ``soundfile``. Ресемплирование к 22050 Гц нужно, чтобы окно 8192 отсчётов
    всегда давало одинаковое частотное разрешение.

    Args:
        path: Путь к файлу ``.wav``, ``.mp3``, ``.flac`` и т. п.
        sample_rate: Целевая частота дискретизации в герцах.

    Returns:
        Кортеж ``(y, sr)``: одномерный сигнал ``float32`` и фактическая частота
        дискретизации.

    Raises:
        FileNotFoundError: Если файл не существует.
        ValueError: Если файл пустой, частота дискретизации неположительная
            или аудио не удалось прочитать.
    """
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
    if sample_rate <= 0:
        raise ValueError("sample_rate должен быть положительным")

    try:
        y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    except Exception as error:  # librosa/audioread/soundfile бросают разные типы
        raise ValueError(f"Не удалось прочитать аудиофайл {audio_path}: {error}") from error

    if y.size == 0:
        raise ValueError(f"Аудиофайл пустой: {audio_path}")

    return np.asarray(y, dtype=np.float32), int(sr)


def estimate_tuning(
    y: np.ndarray,
    sr: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> float:
    """Оценить отклонение строя трека от стандарта A4 = 440 Гц.

    Функция ``librosa.estimate_tuning`` сравнивает спектральные пики с
    идеальной сеткой нот и возвращает поправку в полутонах. Эта поправка
    вычитается из MIDI-номера в :func:`core.fourier.hz_to_midi`, чтобы ноты
    попадали в правильные классы высоты тона.

    Args:
        y: Одномерный аудиосигнал.
        sr: Частота дискретизации в герцах.
        n_fft: Размер окна для оценки спектра.
        hop_length: Шаг между окнами.

    Returns:
        Поправка в полутонах (например, ``0.25`` — трек на четверть тона выше).
        При ошибке оценки возвращается ``0.0``.
    """
    try:
        tuning = librosa.estimate_tuning(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    except Exception:
        return 0.0

    if not np.isfinite(tuning):
        return 0.0
    return float(tuning)


def harmonic_percussive_split(
    magnitude: np.ndarray,
    kernel_size: int = 31,
    margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Разделить спектрограмму на гармоническую и ударную части (HPSS).

    Гармонические звуки (ноты) в спектрограмме выглядят как горизонтальные
    линии: одна и та же частота держится несколько кадров. Ударные (барабаны)
    выглядят как вертикальные всплески по всем частотам. Медианные фильтры
    вдоль времени и вдоль частоты позволяют построить мягкие маски и разделить
    спектр на две части.

    Args:
        magnitude: Неотрицательная спектрограмма (модуль STFT) формы
            ``(n_freq, n_frames)``.
        kernel_size: Размер медианного фильтра (нечётное число). Больше —
            грубее разделение, но устойчивее к шуму.
        margin: Запас разделения: чем больше, тем меньше энергии попадает в обе
            части одновременно.

    Returns:
        Кортеж ``(harmonic, percussive)`` — две неотрицательные матрицы той же
        формы, что и ``magnitude``.

    Raises:
        ValueError: Если спектрограмма не двумерная, содержит отрицательные
            значения или ``kernel_size`` чётный/неположительный.
    """
    spectrum = np.asarray(magnitude, dtype=np.float32)
    if spectrum.ndim != 2:
        raise ValueError("magnitude должна быть двумерной матрицей")
    if np.any(spectrum < 0):
        raise ValueError("magnitude не может содержать отрицательные значения")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size должен быть положительным нечётным числом")
    if margin <= 0:
        raise ValueError("margin должен быть положительным")

    harmonic, percussive = librosa.decompose.hpss(
        spectrum,
        kernel_size=kernel_size,
        margin=margin,
    )
    return harmonic.astype(np.float32), percussive.astype(np.float32)


def spectrum_to_chroma(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    tuning: float = 0.0,
    fmin: float = C2_HZ,
    fmax: float = DEFAULT_FMAX_HZ,
    power: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """Свернуть спектр в хроматограмму из 12 классов высоты тона.

    Каждый частотный бин переводится в MIDI-ноту по формуле
    ``69 + 12 * log2(f / 440)``. Затем номер ноты берётся по модулю 12, поэтому
    все октавы одной ноты (C2, C3, C4, ...) складываются в один класс C.
    Энергия бина добавляется в соответствующий столбец хроматограммы.

    Args:
        magnitude: Спектрограмма амплитуд формы ``(n_freq, n_frames)``.
        frequencies: Частоты строк спектрограммы в герцах, форма ``(n_freq,)``.
        tuning: Поправка строя в полутонах из :func:`estimate_tuning`.
        fmin: Нижняя граница диапазона (по умолчанию C2, около 65 Гц).
        fmax: Верхняя граница диапазона (по умолчанию 2 кГц).
        power: Степень, в которую возводится амплитуда перед суммированием.
            ``2.0`` соответствует энергии сигнала.
        normalize: Если ``True``, каждый столбец нормируется по L2, чтобы
            громкость кадра не влияла на форму хроматограммы.

    Returns:
        Матрица ``float32`` формы ``(12, n_frames)`` со значениями от 0 до 1
        при ``normalize=True``.

    Raises:
        ValueError: Если размеры не согласованы или границы диапазона заданы
            неверно.
    """
    spectrum = np.asarray(magnitude, dtype=np.float64)
    freq_values = np.asarray(frequencies, dtype=np.float64)

    if spectrum.ndim != 2:
        raise ValueError("magnitude должна быть двумерной матрицей")
    if freq_values.ndim != 1 or freq_values.shape[0] != spectrum.shape[0]:
        raise ValueError(
            "frequencies должна содержать по одному значению на строку magnitude"
        )
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Должно выполняться 0 < fmin < fmax")
    if power <= 0:
        raise ValueError("power должен быть положительным")

    chroma = np.zeros((12, spectrum.shape[1]), dtype=np.float64)
    valid = (
        np.isfinite(freq_values)
        & (freq_values >= fmin)
        & (freq_values <= fmax)
        & (freq_values > 0)
    )
    if not np.any(valid):
        return chroma.astype(np.float32)

    midi_numbers = hz_to_midi(freq_values[valid], tuning=tuning)
    pitch_classes = np.mod(np.rint(midi_numbers).astype(np.int64), 12)
    energies = np.power(np.maximum(spectrum[valid], 0.0), power)

    for pitch_class in range(12):
        rows = energies[pitch_classes == pitch_class]
        if rows.size:
            chroma[pitch_class] = rows.sum(axis=0)

    if normalize:
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        chroma = np.divide(
            chroma,
            norms,
            out=np.zeros_like(chroma),
            where=norms > 0,
        )

    return chroma.astype(np.float32)


def compute_chromagram(
    transform: STFT,
    tuning: float = 0.0,
    fmin: float = C2_HZ,
    fmax: float = DEFAULT_FMAX_HZ,
    power: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """Построить обычную хроматограмму по результату STFT.

    Args:
        transform: Результат :func:`core.fourier.stft`.
        tuning: Поправка строя в полутонах.
        fmin: Нижняя граница частот (C2).
        fmax: Верхняя граница частот (2 кГц).
        power: Степень амплитуды при суммировании.
        normalize: Нормировать ли каждый кадр по L2.

    Returns:
        Хроматограмма формы ``(12, n_frames)``.
    """
    return spectrum_to_chroma(
        transform.magnitude,
        transform.frequencies,
        tuning=tuning,
        fmin=fmin,
        fmax=fmax,
        power=power,
        normalize=normalize,
    )


def compute_bass_chromagram(
    transform: STFT,
    tuning: float = 0.0,
    fmin: float = C2_HZ,
    fmax: float = C4_HZ,
    power: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """Построить басовую хроматограмму по низким частотам.

    Диапазон C2–C4 (65–260 Гц) содержит основные тоны баса и низких нот
    гитары. Эта хроматограмма позже помогает отличить, например, C от Am7:
    у C в басу звучит C, у Am7 — A.

    Args:
        transform: Результат :func:`core.fourier.stft`.
        tuning: Поправка строя в полутонах.
        fmin: Нижняя граница (C2).
        fmax: Верхняя граница (C4).
        power: Степень амплитуды при суммировании.
        normalize: Нормировать ли каждый кадр по L2.

    Returns:
        Базовая хроматограмма формы ``(12, n_frames)``.
    """
    return spectrum_to_chroma(
        transform.magnitude,
        transform.frequencies,
        tuning=tuning,
        fmin=fmin,
        fmax=fmax,
        power=power,
        normalize=normalize,
    )


def analyze_audio(
    path: PathLike,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = C2_HZ,
    fmax: float = DEFAULT_FMAX_HZ,
    bass_fmax: float = C4_HZ,
    tuning: Optional[float] = None,
    hpss_kernel_size: int = 31,
    hpss_margin: float = 1.0,
    power: float = 2.0,
) -> AudioAnalysis:
    """Выполнить полный анализ трека для задачи 1.

    Шаги:

    1. Загрузка аудио и приведение к моно/22050 Гц.
    2. Оценка строя через :func:`estimate_tuning`.
    3. Собственный STFT на ``numpy.fft.rfft``.
    4. HPSS по модулю STFT: гармоническая и ударная части.
    5. Хроматограмма гармонической части (65 Гц – 2 кГц).
    6. Базовая хроматограмма (65–260 Гц).
    7. Спектрограмма в дБ для графика.

    Args:
        path: Путь к аудиофайлу.
        sample_rate: Целевая частота дискретизации.
        n_fft: Размер окна STFT (по умолчанию 8192).
        hop_length: Шаг между окнами (по умолчанию 2048).
        fmin: Нижняя граница обычной хроматограммы (C2).
        fmax: Верхняя граница обычной хроматограммы (2 кГц).
        bass_fmax: Верхняя граница басовой хроматограммы (C4).
        tuning: Готовая поправка строя. Если ``None``, оценивается автоматически.
        hpss_kernel_size: Размер медианного фильтра HPSS.
        hpss_margin: Запас разделения HPSS.
        power: Степень амплитуды при построении хромы.

    Returns:
        Заполненный :class:`AudioAnalysis`.

    Raises:
        FileNotFoundError: Если аудиофайл не найден.
        ValueError: Если параметры некорректны или аудио не читается.
    """
    y, sr = load_audio(path, sample_rate=sample_rate)

    if tuning is None:
        tuning_value = estimate_tuning(y, sr, n_fft=n_fft, hop_length=hop_length)
    else:
        tuning_value = float(tuning)
        if not np.isfinite(tuning_value):
            raise ValueError("tuning должен быть конечным числом")

    transform = stft(
        y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )
    magnitude = transform.magnitude
    harmonic, percussive = harmonic_percussive_split(
        magnitude,
        kernel_size=hpss_kernel_size,
        margin=hpss_margin,
    )

    chroma = spectrum_to_chroma(
        harmonic,
        transform.frequencies,
        tuning=tuning_value,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )
    bass_chroma = spectrum_to_chroma(
        harmonic,
        transform.frequencies,
        tuning=tuning_value,
        fmin=fmin,
        fmax=bass_fmax,
        power=power,
    )
    spectrogram_db = transform.to_db()

    return AudioAnalysis(
        path=str(path),
        sample_rate=sr,
        duration=float(y.size) / sr,
        tuning=tuning_value,
        spectrogram_db=spectrogram_db,
        harmonic_magnitude=harmonic,
        percussive_magnitude=percussive,
        chroma=chroma,
        bass_chroma=bass_chroma,
        frequencies=transform.frequencies,
        times=transform.times,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов для ручной проверки анализа из консоли.

    Returns:
        Настроенный :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Анализ аудио: STFT, HPSS и хроматограммы (задача 1).",
    )
    parser.add_argument("audio", help="Путь к аудиофайлу (.wav, .mp3, ...)")
    parser.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Частота дискретизации (по умолчанию {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=DEFAULT_N_FFT,
        help=f"Размер окна STFT (по умолчанию {DEFAULT_N_FFT})",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help=f"Шаг между окнами (по умолчанию {DEFAULT_HOP_LENGTH})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа консольной проверки.

    Печатает основные сведения о треке и пять самых сильных нот.

    Args:
        argv: Аргументы командной строки без имени программы. Если ``None``,
            берутся ``sys.argv[1:]``.

    Returns:
        Код возврата: ``0`` при успехе.
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    analysis = analyze_audio(
        args.audio,
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    mean = analysis.mean_chroma()
    order = np.argsort(mean)[::-1][:5]
    notes = ", ".join(f"{PITCH_CLASS_NAMES[int(i)]}={mean[i]:.3f}" for i in order)

    print(f"Файл: {analysis.path}")
    print(f"Частота дискретизации: {analysis.sample_rate} Гц")
    print(f"Длительность: {analysis.duration:.2f} с")
    print(f"Поправка строя: {analysis.tuning:+.2f} полутона")
    print(f"Кадров STFT: {analysis.n_frames}, окно: {analysis.n_fft}, шаг: {analysis.hop_length}")
    print(f"Спектрограмма: {analysis.spectrogram_db.shape}")
    print(f"Хроматограмма: {analysis.chroma.shape}")
    print(f"Басовая хроматограмма: {analysis.bass_chroma.shape}")
    print(f"Самые сильные ноты: {notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
