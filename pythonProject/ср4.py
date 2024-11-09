import numpy as np
import pandas as pd

# Завантаження даних
data = np.genfromtxt('A3.txt', delimiter=',')
data = data.reshape(5000, 12)  # Перетворення даних в форму (5000 рядків, 12 каналів)

# Нормалізація даних
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized_data = (data - mean) / std

# Обчислення кореляційної матриці
correlation_matrix = np.corrcoef(normalized_data, rowvar=False)

# Перетворення в DataFrame для зручного відображення
correlation_df = pd.DataFrame(correlation_matrix,
                              columns=[f'Канал {i+1}' for i in range(12)],
                              index=[f'Канал {i+1}' for i in range(12)])

# Виведення таблиці
print("Кореляційна матриця:")
print(correlation_df)

# Збереження таблиці в файл Excel для звіту
correlation_df.to_excel('correlation_matrix.xlsx', index=True)

# Крок 3: Пошук пар з високою кореляцією (коефіцієнт > 0.9 або < -0.9)
high_corr_pairs = []
threshold = 0.9
for i in range(len(correlation_matrix)):
    for j in range(i):
        if abs(correlation_matrix[i, j]) > threshold:
            high_corr_pairs.append((f'Канал {i+1}', f'Канал {j+1}'))

print("\nПари з високою кореляцією:")
for pair in high_corr_pairs:
    print(pair)

# Функція для обчислення часткового коефіцієнта кореляції
def partial_corr(r_ab, r_ac, r_bc):
    return (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac**2) * (1 - r_bc**2))

# Крок 5: Часткові коефіцієнти кореляції
# 1. Між ознаками a та c без урахування впливу ознаки b
a, b, c = 0, 1, 2  # Канали для аналізу (a, b, c)
r_ab = correlation_matrix[a, b]
r_ac = correlation_matrix[a, c]
r_bc = correlation_matrix[b, c]

r_ac_b = partial_corr(r_ac, r_ab, r_bc)
print(f"\nЧастковий коефіцієнт кореляції між Каналом {a+1} та Каналом {c+1}, без урахування Канала {b+1}: {r_ac_b}")

# 2. Між ознаками a та b без урахування впливу ознак c та d
d = 3  # Канал для d
r_ad = correlation_matrix[a, d]
r_bd = correlation_matrix[b, d]
r_cd = correlation_matrix[c, d]

r_ab_cd = partial_corr(r_ab, r_ac, r_bd)
print(f"\nЧастковий коефіцієнт кореляції між Каналом {a+1} та Каналом {b+1}, без урахування Каналів {c+1} та {d+1}: {r_ab_cd}")

# 3. Між ознаками a та c без урахування впливу ознак b та d
r_ac_bd = partial_corr(r_ac, r_ab, r_bd)
print(f"\nЧастковий коефіцієнт кореляції між Каналом {a+1} та Каналом {c+1}, без урахування Каналів {b+1} та {d+1}: {r_ac_bd}")

# 4. Між ознаками a та d без урахування впливу ознак b та c
r_ad_bc = partial_corr(r_ad, r_ac, r_bc)
print(f"\nЧастковий коефіцієнт кореляції між Каналом {a+1} та Каналом {d+1}, без урахування Каналів {b+1} та {c+1}: {r_ad_bc}")

# Крок 6: Множинний коефіцієнт кореляції для канала a, при лінійному двофакторному зв'язку з b та c
def multiple_corr(r_ab, r_ac, r_bc):
    return np.sqrt((r_ab**2 + r_ac**2 - 2 * r_ab * r_ac * r_bc) / (1 - r_bc**2))

r_a_bc = multiple_corr(r_ab, r_ac, r_bc)
print(f"\nМножинний коефіцієнт кореляції для Канала {a+1} з Каналом {b+1} та Каналом {c+1}: {r_a_bc}")

# Крок 7: Множинний коефіцієнт кореляції для канала a, при лінійному трифакторному зв'язку з b, c та d
def multiple_corr_3factors(r_ab, r_ac, r_ad, r_bc, r_bd, r_cd):
    return np.sqrt(1 - (1 - r_ab**2) * (1 - r_ac**2) * (1 - r_ad**2))

r_a_bcd = multiple_corr_3factors(r_ab, r_ac, correlation_matrix[0, 3], r_bc, correlation_matrix[1, 3], correlation_matrix[2, 3])
print(f"\nТрифакторний множинний коефіцієнт кореляції для Канала {a+1}, Канала {b+1}, Канала {c+1} та Канала {d+1}: {r_a_bcd}")

# Крок 8: Визначення незалежних параметрів
independent_params = []
for col in range(data.shape[1]):
    if all(abs(correlation_matrix[col, other_col]) < 0.3 for other_col in range(data.shape[1]) if other_col != col):
        independent_params.append(f'Канал {col+1}')

print("\nНезалежні канали (кореляція з іншими факторами < 0.3):")
print(independent_params)
