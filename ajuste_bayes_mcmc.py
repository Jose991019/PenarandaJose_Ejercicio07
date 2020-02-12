import numpy as np
import matplotlib.pyplot as plt
datos = np.loadtxt('notas_andes.dat')
x = datos[:,0:4]
y = datos[:,4]
sigma = 0.1

def model(X,B,B0):
    return np.sum(X*B,axis=1) + B0

def loglikelihood(x_obs, y_obs, sigma_y_obs, B, B0):
    d = y_obs -  model(x_obs, B, B0)
    d = d/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d

N = 50000
B = [[np.random.random(),np.random.random(),np.random.random(),np.random.random()]]
B0 = [np.random.random()]
logposterior = [loglikelihood(x, y, sigma, B[0], B0[0])]

sigma_delta_B = 0.1
sigma_delta_B0 = 0.1

for i in range(1,N):
    propuesta_B  = B[i-1] + np.random.randn(1,4)*sigma_delta_B
    propuesta_B0  = B0[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_B0)

    logposterior_viejo = loglikelihood(x, y, sigma, B[i-1], B0[i-1])
    logposterior_nuevo = loglikelihood(x, y, sigma, propuesta_B, propuesta_B0) 

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        B.append(list(propuesta_B)[0])
        B0.append(propuesta_B0)
        logposterior.append(logposterior_nuevo)
    else:
        B.append(B[i-1])
        B0.append(B0[i-1])
        logposterior.append(logposterior_viejo)
B = np.array(B)
B0 = np.array(B0)
logposterior = np.array(logposterior)


sigma_b0 = np.std(B0[10000:])
sigma_b1 = np.std(B[10000:,0])
sigma_b2 = np.std(B[10000:,1])
sigma_b3 = np.std(B[10000:,2])
sigma_b4 = np.std(B[10000:,3])

mean_b0 = np.std(B0[10000:])
mean_b1 = np.std(B[10000:,0])
mean_b2 = np.std(B[10000:,1])
mean_b3 = np.std(B[10000:,2])
mean_b4 = np.std(B[10000:,3])

plt.figure(figsize=(15,15))
plt.subplot(3,2,1)
plt.hist(B0[10000:],density = True,bins = 20)
plt.title("{} = {:.2f} {} {:.2f}".format("$beta_0$",mean_b0,"$\pm$",sigma_b0))
#plt.xlabel("{}".format("$\beta_0$"))
plt.subplot(3,2,2)
plt.hist(B[10000:,0],density = True,bins = 20)
plt.title("{} = {:.2f} {} {:.2f}".format("$beta_1$",mean_b1,"$\pm$",sigma_b1))
plt.subplot(3,2,3)
plt.hist(B[10000:,1],density = True,bins = 20)
plt.title("{} = {:.2f} {} {:.2f}".format("$beta_2$",mean_b2,"$\pm$",sigma_b2))
plt.subplot(3,2,4)
plt.hist(B[10000:,2],density = True,bins = 20)
plt.title("{} = {:.2f} {} {:.2f}".format("$beta_3$",mean_b3,"$\pm$",sigma_b3))
plt.subplot(3,2,5)
plt.hist(B[10000:,3],density = True,bins = 20)
plt.title("{} = {:.2f} {} {:.2f}".format("$beta_4$",mean_b4,"$\pm$",sigma_b4))

plt.savefig("ajuste_bayes_mcmc.png")