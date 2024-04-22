import numpy as np
import statsmodels.api as sm

# Параметры моделирования
N = 100000
sigma = 1

# Значения rho от 0.5 до 0.95 с шагом 0.05
rhos = np.arange(0.5, 1.0, 0.05)

# Генерация начальных значений
def generate_initial_values():
    U1 = np.random.rand()
    U2 = np.random.rand()
    X1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    X2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return X1, X2

# Генерация временного ряда
def generate_sequence(N, rho, sigma):
    r = np.sqrt(rho)
    ksi1 = np.zeros(N)
    ksi2 = np.zeros(N)
    tau = np.zeros(N)
    
    x1_prev, x2_prev = generate_initial_values()
    ksi1[0] = sigma * np.power((np.sqrt(1 - rho) * np.random.randn() + r * x1_prev), 2)
    ksi2[0] = sigma * np.power((np.sqrt(1 - rho) * np.random.randn() + r * x2_prev), 2)
    tau[0] = ksi1[0] + ksi2[0]
    
    for i in range(1, N):
        x1 = np.sqrt(1 - rho) * np.random.randn() + r * x1_prev
        x2 = np.sqrt(1 - rho) * np.random.randn() + r * x2_prev
        
        ksi1[i] = sigma * np.power((np.sqrt(1 - rho) * np.random.randn() - r * np.sqrt(ksi1[i - 1])), 2)
        ksi2[i] = sigma * np.power((np.sqrt(1 - rho) * np.random.randn() - r * np.sqrt(ksi2[i - 1])), 2)
        tau[i] = ksi1[i] + ksi2[i]
        
        x1_prev, x2_prev = x1, x2
        
    return tau

for rho in rhos:
    print(f"Рассчитывается для rho = {rho}")
    tau_sequence = generate_sequence(N, rho, sigma)
    print(sm.tsa.acf(tau_sequence, nlags=2))
    print()
