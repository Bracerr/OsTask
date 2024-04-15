import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Параметры моделирования
sigma_0_sq = 0.25
rho_1_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
num_steps = 100000

def generate_normal():
    U1 = np.random.rand()
    U2 = np.random.rand()
    X1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    X2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return X1, X2

def generate_tau_sequence(rho_1):
    tau_sequence = np.zeros(num_steps)
    
    # Начальные значения xi1_prev_sq и xi2_prev_sq
    X1_prev, X2_prev = generate_normal()
    xi1_prev_sq = sigma_0_sq * (np.sqrt(1 - rho_1) * X1_prev)**2
    xi2_prev_sq = sigma_0_sq * (np.sqrt(1 - rho_1) * X2_prev)**2
    
    for n in range(num_steps):
        # Моделирование xi1_sq и xi2_sq
        X1, X2 = generate_normal()
        xi1_sq = sigma_0_sq * ((np.sqrt(1 - rho_1) * X1) + np.sqrt(rho_1) * np.sqrt(xi1_prev_sq))**2
        xi2_sq = sigma_0_sq * ((np.sqrt(1 - rho_1) * X2) + np.sqrt(rho_1) * np.sqrt(xi2_prev_sq))**2
        
        # Вычисление tau(n)
        tau_sequence[n] = xi1_sq + xi2_sq
        
        # Обновление значений для следующего шага
        xi1_prev_sq = xi1_sq
        xi2_prev_sq = xi2_sq
    
    return tau_sequence

# Моделирование и вычисление корреляций
for rho_1 in rho_1_values:
    tau_sequence = generate_tau_sequence(rho_1)
    
    # Вычисление коэффициента корреляции
    corr_coef, _ = pearsonr(tau_sequence[:-1], tau_sequence[1:])
    print(f"Экспериментальный коэффициент корреляции для rho_1 = {rho_1}: {corr_coef}")
    
    # Визуализация последовательности
    plt.figure(figsize=(10, 4))
    plt.plot(tau_sequence[:1000])  # Показываем первые 1000 отсчетов для наглядности
    plt.title(f"Последовательность tau(n) для rho_1 = {rho_1}")
    plt.xlabel("n")
    plt.ylabel("tau(n)")
    plt.show()
