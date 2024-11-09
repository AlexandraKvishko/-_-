import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hmean, gmean, skew, kurtosis, shapiro

# Параметри завдання
N = 5000  # кількість точок у файлі
channels = 12  # кількість каналів
sampling_rate = 500  # частота дискретизації
time = np.linspace(0, N / sampling_rate, N)

# Завантаження даних з одного файлу
file = "A3.txt"
data = pd.read_csv(file, sep=",", header=None)

# Перевірка структури даних
print("Структура даних:")
print(data.head())  # Виводимо перші кілька рядків для перевірки

# Заміна ком на крапки для правильної інтерпретації чисел
data = data.apply(lambda x: x.map(lambda y: str(y).replace(",", ".")))

# Перетворення даних у числовий формат
data = data.apply(pd.to_numeric, errors='coerce')

# Видалення рядків з NaN (якщо такі є)
data = data.dropna()

# Перевірка розміру даних
print(f"Розмір даних: {data.shape}")

# Ініціалізація списку для збереження статистичних параметрів кожного каналу
stats_summary = []

# Обробка кожного каналу
for channel in range(channels):
    if channel >= data.shape[1]:
        print(f"Канал {channel + 1} відсутній у даних.")
        continue

    channel_data = data[channel]

    # Перевірка, чи не порожні дані
    if channel_data.empty:
        print(f"Канал {channel + 1} порожній, пропускаємо.")
        continue

    # Розрахунок статистичних параметрів
    mean_value = channel_data.mean()
    hmean_value = hmean(channel_data[channel_data > 0]) if (
                channel_data[channel_data > 0].size > 0) else np.nan  # Гармонійне середнє (без нулів)
    gmean_value = gmean(channel_data[channel_data > 0]) if (
                channel_data[channel_data > 0].size > 0) else np.nan  # Геометричне середнє
    variance_value = channel_data.var()
    std_dev_value = channel_data.std()  # Середньоквадратичне відхилення
    gini_diff = channel_data.diff().abs().mean()

    # Обробка моди з використанням pandas .mode() з перевіркою на наявність значень
    most_frequent_value = channel_data.mode().iloc[0] if not channel_data.mode().empty else np.nan

    # Знаходимо медіану
    median_value = channel_data.median()
    min_median_value = channel_data[channel_data == channel_data.min()].median()  # Найменша медіана
    skewness_value = skew(channel_data) if len(channel_data) > 1 else np.nan  # Перевірка на наявність даних для розрахунку асиметрії
    kurtosis_value = kurtosis(channel_data) if len(channel_data) > 1 else np.nan  # Перевірка на наявність даних для розрахунку ексцесу

    # Додавання статистики каналу у список
    stats_summary.append({
        'Канал': channel + 1,
        'Середнє': mean_value,
        'Гармонійне середнє': hmean_value,
        'Геометричне середнє': gmean_value,
        'Дисперсія': variance_value,
        'Середнє квадратичне відхилення': std_dev_value,
        'Найменша медіана': min_median_value,
        'Мода': most_frequent_value,
        'Медіана': median_value,
        'Асиметрія': skewness_value,
        'Ексцес': kurtosis_value
    })

    # Перевірка на нормальність за допомогою тесту Шапіро-Вілка
    stat, p_value = shapiro(channel_data)
    stats_summary[-1]['p-значення нормальності'] = p_value

    # Нормалізація даних для кожного каналу
    normalized_data = (channel_data - mean_value) / np.std(channel_data)
    data[channel] = normalized_data

    # Побудова графіка для кожного каналу
    plt.figure(figsize=(10, 4))
    plt.plot(time, normalized_data[:N], label=f"Канал {channel + 1}")
    plt.xlabel("Час (с)")
    plt.ylabel("Нормалізована амплітуда")
    plt.title(f"ЕКГ Канал {channel + 1}")
    plt.legend()
    plt.show()

    # Побудова гістограми даних
    plt.figure(figsize=(8, 4))
    plt.hist(channel_data, bins=30, alpha=0.7, label=f"Канал {channel + 1}")
    plt.xlabel("Амплітуда")
    plt.ylabel("Частота")
    plt.title(f"Гістограма Каналу {channel + 1}")
    plt.legend()
    plt.show()

# Створення DataFrame зі статистиками для всіх каналів
df_stats = pd.DataFrame(stats_summary)

# Збереження таблиці в Excel
output_file = "EKG_Statistics.xlsx"
df_stats.to_excel(output_file, index=False)

# Виведення повідомлення про успішне збереження
print(f"Таблиця статистики збережена в {output_file}")

# Побудова графіка перших 6 каналів
plt.figure(figsize=(12, 6))
for i in range(6):
    if i < data.shape[1]:
        plt.plot(time, data[i][:N], label=f"Канал {i + 1}")
    else:
        print(f"Канал {i + 1} відсутній у даних.")
plt.xlabel("Час (с)")
plt.ylabel("Нормалізована амплітуда")
plt.title("ЕКГ - Перші 6 Каналів")
plt.legend()
plt.show()
