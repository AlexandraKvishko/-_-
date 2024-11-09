import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження даних
data = np.genfromtxt('A3.txt', delimiter=',')
data = data.reshape(5000, 12)  # Перетворення даних у форму (5000 рядків, 12 каналів)

# Побудова кореляційної матриці для всіх 12 каналів.
R = np.corrcoef(data.T)

# Крок 1: Знаходження власних чисел і частка дисперсії
# Обчислюємо власні числа та власні вектори для кореляційної матриці.
eigenvalues, eigenvectors = np.linalg.eig(R)

# Сортуємо власні числа за спаданням, щоб знайти найбільші компоненти
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Розрахунок частки дисперсії для кожної компоненти
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
cumulative_variance = np.cumsum(explained_variance)

# Крок 2: Побудова графіка критерію кам’янистого осипу
# Це графік власних чисел для визначення значущих компонент (кам’янистий осип).
plt.figure()
plt.plot(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, 'o-', label="Власні значення")
plt.axhline(y=1, color='r', linestyle='--', label="Лінія на рівні 1")  # Добавляем горизонтальную линию на уровне 1
plt.xlabel("Компонента")
plt.ylabel("Власне значення")
plt.title("Критерій кам'янистого осипу")
plt.legend()
plt.grid()
plt.show()

# Обчислення інформативності кожної компоненти (I_k)
# Це значення, що відображає вклад кожної компоненти в загальну дисперсію.
informative_index = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
print("\nІнформативність компонент:")
print(informative_index)

# Таблиця 1 - Власні числа та частка дисперсії
table1 = pd.DataFrame({
    "№п/п": np.arange(1, len(sorted_eigenvalues) + 1),
    "Власні числа": sorted_eigenvalues,
    "Частка дисперсії": explained_variance,
    "Сумарна дисперсія": cumulative_variance
})
print("Таблиця 1: Власні числа та частка дисперсії")
print(table1)

# Крок 3: Власні вектори (матриця L)
L = sorted_eigenvectors  # Матриця власних векторів
table2 = pd.DataFrame(L, columns=[f"Компонента {i + 1}" for i in range(L.shape[1])])
print("\nТаблиця 2: Власні вектори (матриця L)")
print(table2)

# Таблиця 3 - Власний вектор для максимального власного числа
# Знайдемо власний вектор для максимального власного числа (першого вектора).
principal_vector = L[:, 0]  # Вектор максимального власного числа
table3 = pd.DataFrame({"Власний вектор максимального власного числа": principal_vector})
print("\nТаблиця 3: Власний вектор максимального власного числа")
print(table3)

# Крок 5: Знаходження головних компонент
principal_components = data.dot(L)  # Матриця головних компонент

# Крок 4: Перевірка ортогональності власних векторів
# Перевіряємо, чи є власні вектори ортогональними, тобто чи їх скалярний добуток дорівнює нулю для i != j.
print("\nПеревірка ортогональності власних векторів (a'_j * a_k):")
for i in range(len(sorted_eigenvectors)):
    for j in range(i + 1, len(sorted_eigenvectors)):
        dot_product = np.dot(sorted_eigenvectors[:, i], sorted_eigenvectors[:, j])
        print(f"Скалярний добуток a_{i + 1}' та a_{j + 1}: {dot_product}")

# Перевірка норм власних векторів (a'_k * a_k = 1)
# Перевіряємо, чи норма кожного власного вектора дорівнює 1.
print("\nПеревірка норм власних векторів (a'_k * a_k = 1):")
for i in range(len(sorted_eigenvectors)):
    norm = np.linalg.norm(sorted_eigenvectors[:, i])
    print(f"Норма власного вектора a_{i + 1}: {norm}")

# Таблиця 8: Перші три головні фактори
table4 = pd.DataFrame({
    "№п/п": np.arange(1, principal_components.shape[0] + 1),
    "Z1": principal_components[:, 0],
    "Z2": principal_components[:, 1],
    "Z3": principal_components[:, 2]
})
print("\nТаблиця 4: Перші три головні фактори")
print(table4)

# Графік першої головної компоненти
plt.figure(figsize=(10, 6))
plt.plot(table4["№п/п"], table4["Z1"], label="Перша головна компонента (Z1)", color="b")
plt.xlabel("Час")
plt.ylabel("Значення компоненти Z1")
plt.title("Графік першої головної компоненти")
plt.legend()
plt.grid()
plt.show()

# Крок 6: Перевірка умов для головних компонент: сума всіх значень головної компоненти повинна дорівнювати нулю.
for i in range(principal_components.shape[1]):
    sum_z = np.sum(principal_components[:, i])
    print(f"Сума компонент Z{i + 1}: {sum_z}")

# Перевірка дисперсії: 1/n * z'_k * z_k = лямда_k
for i in range(principal_components.shape[1]):
    variance_check = np.mean(principal_components[:, i] ** 2)
    print(f"Перевірка для компоненти Z{i + 1}: {variance_check} ≈ {sorted_eigenvalues[i]}")

# Перевірка ортогональності головних компонент: z'_j * z_k = 0 для j != k
for i in range(principal_components.shape[1]):
    for j in range(i + 1, principal_components.shape[1]):
        dot_product = np.dot(principal_components[:, i], principal_components[:, j])
        print(f"Скалярний добуток Z_{i + 1} та Z_{j + 1}: {dot_product}")

# Збереження всіх таблиць в Excel
with pd.ExcelWriter("Результати_факторного_аналізу.xlsx") as writer:
    table1.to_excel(writer, sheet_name="Таблиця 1 - Власні числа", index=False)
    table2.to_excel(writer, sheet_name="Таблиця 2 - Власні вектори", index=False)
    table3.to_excel(writer, sheet_name="Таблиця 3 - Макс. власн. вектор", index=False)
    table4.to_excel(writer, sheet_name="Таблиця 4 - Головні фактори", index=False)

print("Усі таблиці збережено у файл 'Результати_факторного_аналізу.xlsx'")
