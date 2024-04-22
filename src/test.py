import math
import random
import statsmodels.api as sm

N = 100000
rho = 0.8 #коэффициент корреляции
r = math.sqrt(rho)
sigma = 1

ksi1 = [None]*N
ksi2 = [None]*N
tau = [None]*N

ksi1[0] = sigma * math.pow((math.sqrt(1 - rho) * random.normalvariate(0,sigma)),2)
ksi2[0] = sigma * math.pow((math.sqrt(1 - rho) * random.normalvariate(0,sigma)),2)
tau[0] = ksi1[0] + ksi2[0]

for i in range(1,N):
    ksi1[i] = sigma * math.pow((math.sqrt(1 - rho) * random.normalvariate(0,sigma) - r * math.sqrt(ksi1[i - 1])),2)
    ksi2[i] = sigma * math.pow((math.sqrt(1 - rho) * random.normalvariate(0,sigma) - r * math.sqrt(ksi2[i - 1])),2)
    tau[i] = ksi1[i] + ksi2[i]

print (sm.tsa.acf(tau, nlags = 2))