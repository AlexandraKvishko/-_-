import numpy as np
import matplotlib.pyplot as plt

# Завантаження даних
data = np.genfromtxt('A3.txt', delimiter=',')
data = data.reshape(5000, 12)  # Розділяємо на 12 каналів

N = 5000  # кількість точок
t = np.linspace(0, 10, N)  # Час запису 10 секунд
channels = data.T  # Кожен канал як окремий рядок (12 x 5000)

# Крок по частоті
freq_step = 1 / 10  # Δf = 1/T = 1/10 = 0.1 Гц
print(f"Частота першої синусоїди (крок по частоті) = {freq_step} Гц")

# Обчислення коефіцієнтів Фур'є для кожного каналу
A = []
B = []
C = []

for channel in channels:
    # Розрахунок A_j і B_j
    A_j = np.zeros(N // 2 + 1)
    B_j = np.zeros(N // 2 + 1)

    for j in range(N // 2 + 1):
        if j == 0:
            A_j[j] = (1 / N) * np.sum(channel * np.cos((2 * np.pi * np.arange(N) * 0) / N))
        elif j == N // 2:
            A_j[j] = (1 / N) * np.sum(channel * np.cos((np.pi * np.arange(N)) / 1))
        else:
            A_j[j] = (2 / N) * np.sum(channel * np.cos((2 * np.pi * np.arange(N) * j) / N))
            B_j[j] = (2 / N) * np.sum(channel * np.sin((2 * np.pi * np.arange(N) * j) / N))

    # Розрахунок спектру C_j
    C_j = np.sqrt(A_j ** 2 + B_j ** 2)
    A.append(A_j)
    B.append(B_j)
    C.append(C_j)

# Перетворюємо A, B, C у масиви для зручності
A = np.array(A)
B = np.array(B)
C = np.array(C)

#Обернене перетворення Фур'є для відновлення сигналу
restored_data = []

for k in range(12):
    restored_signal = np.zeros(N)
    for i in range(N):
        restored_signal[i] = np.sum(A[k] * np.cos((2 * np.pi * np.arange(N // 2 + 1) * i) / N) +
                                    B[k] * np.sin((2 * np.pi * np.arange(N // 2 + 1) * i) / N))
    restored_data.append(restored_signal)

restored_data = np.array(restored_data)

#Спектр модуля сигналу для кожного каналу
for i in range(12):
    plt.figure(figsize=(10, 4))
    plt.plot(C[i])  # Загальний спектр для каналу
    plt.title(f"Рис. 14: Спектр C_j для каналу {i + 1}")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.show()

#Перші 200 точок початкового і відновленого сигналу для кожного каналу
for i in range(12):
    plt.figure(figsize=(12, 6))
    plt.plot(t[:200], channels[i, :200], label='Початковий сигнал')
    plt.plot(t[:200], restored_data[i, :200], label='Відновлений сигнал', linestyle='--')
    plt.title(f"Рис. 15: Порівняння початкового і відновленого сигналу для каналу {i + 1} (перші 200 точок)")
    plt.xlabel("Час (с)")
    plt.ylabel("Амплітуда")
    plt.legend()
    plt.grid(True)
    plt.show()

#Обчислення середньої похибки для кожного каналу
for i in range(12):
    error = np.mean(np.abs(channels[i] - restored_data[i]))
    print(f"Середня похибка для каналу {i + 1}: {error:.10f}")
