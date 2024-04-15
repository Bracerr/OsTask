import numpy as np
import matplotlib.pyplot as plt

# Параметры моделирования
sigma_0_sq = 0.25
rhos = [0.7] + list(np.arange(0.5, 1.0, 0.05))
n_steps = 100000

# Генерация начальных значений x1(0) и x2(0)
def generate_initial_values():
    U1 = np.random.rand()
    U2 = np.random.rand()
    X1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    X2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return X1, X2

# Генерация последовательности
def generate_sequence(n_steps, sigma_0_sq, r_sq):
    x1_prev, x2_prev = generate_initial_values()
    tau_seq = np.zeros(n_steps)
    
    for n in range(n_steps):
        x1 = np.sqrt(1 - r_sq) * np.random.randn() + r_sq * x1_prev
        x2 = np.sqrt(1 - r_sq) * np.random.randn() + r_sq * x2_prev
        
        xi1_sq = sigma_0_sq * (np.sqrt(1 - r_sq) * x1 + r_sq * x1_prev)**2
        xi2_sq = sigma_0_sq * (np.sqrt(1 - r_sq) * x2 + r_sq * x2_prev)**2
        
        tau_seq[n] = xi1_sq + xi2_sq
        
        x1_prev, x2_prev = x1, x2
    
    return tau_seq

# Вычисление коэффициента корреляции
def calculate_correlation(seq):
    return np.corrcoef(seq[:-1], seq[1:])[0, 1]

# Моделирование и вычисление коэффициента корреляции для rho_1 = 0.7
r_sq = rhos[0]
tau_seq = generate_sequence(n_steps, sigma_0_sq, r_sq)
correlation = calculate_correlation(tau_seq)
print(f"Коэффициент корреляции для rho_1 = {r_sq}: {correlation}")

# Моделирование и вычисление коэффициента корреляции для остальных значений rho_1
correlations = []

for rho in rhos[1:]:
    r_sq = rho
    tau_seq = generate_sequence(n_steps, sigma_0_sq, r_sq)
    correlation = calculate_correlation(tau_seq)
    print(f"Коэффициент корреляции для rho_1 = {rho}: {correlation}")
    correlations.append(correlation)

# # Визуализация зависимости коэффициента корреляции от rho
# plt.figure(figsize=(12, 6))
# plt.plot(rhos[1:], correlations, marker='o')
# plt.title('Correlation Coefficient vs rho')
# plt.xlabel('rho')
# plt.ylabel('Correlation Coefficient')
# plt.grid(True)
# plt.show()
