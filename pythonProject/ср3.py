import numpy as np
import pandas as pd
from scipy.stats import f

# Завантаження даних із файлу
data = np.genfromtxt('A3.txt', delimiter=',')
data = data.reshape(5000, 12)

# Параметри задачі
n_channels = 12  # кількість каналів (A)
n_parts = 5      # кількість частин у кожному каналі (B)
n_points = 1000  # кількість значень у кожній частині

# Розбиваємо дані на канали і частини та обчислюємо середні значення для кожної клітинки
channels = data.T  # Транспонуємо, щоб кожен канал був окремим рядком (12 x 5000)
table = [[np.mean(channels[i][j * n_points:(j + 1) * n_points]) for j in range(n_parts)] for i in range(n_channels)]

# Створення DataFrame для експорту в Excel із середніми значеннями
rows = []
for j in range(n_parts):
    row = {f'A{i + 1}': table[i][j] for i in range(n_channels)}
    rows.append(row)

# Перетворюємо список рядків у DataFrame
df = pd.DataFrame(rows)

# Перейменовуємо індекси рядків, щоб відобразити рівні фактора B
df.index = [f'B{j + 1}' for j in range(n_parts)]

# Зберігаємо таблицю з середніми значеннями в Excel
df.to_excel("кардіограма_середні_значення.xlsx", index=True)

# === Розрахунок статистичних показників ===

# Обчислюємо середні значення x_ij для кожної клітинки (вже обчислені раніше)
xij = np.array(table)

# Основні показники
Q1 = np.sum(xij**2)
Q2 = (1 / n_parts) * np.sum(np.sum(xij, axis=1)**2)
Q3 = (1 / n_channels) * np.sum(np.sum(xij, axis=0)**2)
Q4 = (1 / (n_channels * n_parts)) * (np.sum(xij)**2)

# Оцінка дисперсій
S2_0 = (Q1 + Q4 - Q2 - Q3) / ((n_channels - 1) * (n_parts - 1))
S2_A = (Q2 - Q4) / (n_channels - 1)
S2_B = (Q3 - Q4) / (n_parts - 1)

# Виведення оцінок дисперсій
print(f"Оцінка дисперсії S^2_0: {S2_0}")
print(f"Оцінка дисперсії S^2_A: {S2_A}")
print(f"Оцінка дисперсії S^2_B: {S2_B}")

# Перевірка значущості факторів за допомогою критерію Фішера
alpha = 0.05  # рівень значущості
F_a_A = f.ppf(1 - alpha, dfn=n_channels - 1, dfd=(n_channels - 1) * (n_parts - 1))
F_a_B = f.ppf(1 - alpha, dfn=n_parts - 1, dfd=(n_channels - 1) * (n_parts - 1))

is_A_significant = (S2_A / S2_0) > F_a_A
is_B_significant = (S2_B / S2_0) > F_a_B

print("Значущість фактора A:", "Значущий" if is_A_significant else "Не значущий")
print("Значущість фактора B:", "Значущий" if is_B_significant else "Не значущий")

# Якщо фактори A і B залежні, додаємо розрахунок Q5 і S^2_AB
if is_A_significant and is_B_significant:
    Q5 = np.sum([np.sum([np.sum(channels[i][j * n_points:(j + 1) * n_points]**2) for j in range(n_parts)]) for i in range(n_channels)])
    S2_AB = (Q5 - n_points * Q1) / (n_channels * n_parts * (n_points - 1))

    # Перевірка взаємодії факторів
    F_ab = (n_points * S2_0) / S2_AB
    f1 = (n_channels - 1) * (n_parts - 1)
    f2 = n_channels * n_parts * (n_points - 1)
    F_crit_AB = f.ppf(1 - alpha, dfn=f1, dfd=f2)

    is_AB_interaction_significant = F_ab > F_crit_AB
    print("Взаємодія факторів A і B:", "Значуща" if is_AB_interaction_significant else "Не значуща")
