import numpy as np
import pandas as pd
from scipy import stats

# Завантажуємо дані з файлу
data = np.genfromtxt('A3.txt', delimiter=',')
data = data.reshape(5000, 12)  # Перетворюємо дані в форму (5000 точок, 12 каналів)

# Параметри
n = 5000  # Кількість точок
k = 12    # Кількість каналів
alpha = 0.05  # Рівень значущості

# 1. Обчислюємо середнє значення для кожного каналу
means = np.mean(data, axis=0)

# 2. Обчислюємо дисперсії для кожного каналу
S_squared_i = np.var(data, axis=0, ddof=1)

# 3. Перевірка рівності дисперсій:
# Тест Левене на рівність дисперсій
_, p_value = stats.levene(*[data[:, i] for i in range(k)])

print(f"\nP-значення для тесту Левене (перевірка рівності дисперсій): {p_value}")

if p_value > alpha:
    print("Гіпотеза про рівність дисперсій не відкидається.")
else:
    print("Гіпотеза про рівність дисперсій відкидається.")

# 4. Перевірка, чи є статистично значущим вплив каналу (фактора) за допомогою F-статистики

# Загальна дисперсія S_0^2
S_0_squared = np.var(data.flatten(), ddof=k-1)

# Дисперсія, пов'язана з фактором (S_A^2)
mean_all = np.mean(data)  # Загальне середнє по всіх даних
S_A_squared = (n / (k - 1)) * np.sum((means - mean_all) ** 2)

# F-статистика
F_statistic = S_A_squared / S_0_squared

# Критичне значення F для заданого рівня значущості та ступенів свободи
F_critical = stats.f.ppf(1 - alpha, k - 1, k * (n - 1))

print(f"\nF-статистика: {F_statistic}")
print(f"Критичне значення F: {F_critical}")

if F_statistic > F_critical:
    print("Вплив канала (фактора) статистично значущий")
else:
    print("Вплив канала (фактора) не статистично значущий")

# 5. Виводимо результати (середні значення та дисперсії для кожного каналу)
results = pd.DataFrame({
    'Канал': [f'Канал {i+1}' for i in range(k)],
    'Середнє значення': means,
    'Дисперсія': S_squared_i
})

print("\nРезультати для кожного каналу:")
print(results)

# 6. Зберігаємо результати в Excel
results.to_excel('dispersions_analysis_results.xlsx', index=False)
