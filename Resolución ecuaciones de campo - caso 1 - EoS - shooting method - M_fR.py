# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:13:25 2023

@author: Raul
"""

#Solución de las ecuaciones de campo para f(R) = R + alpha*R^2

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.interpolate as si
from scipy.optimize import curve_fit

#p_convert = 1.60218e-13*1e45*10*8.26e-40  #km^-2


#Obtenemos los valores de densidad y presión de los archivos txt
path_P_max = r'C:\Users\Raul\Desktop\Unversidad de Salamanca\TRABAJO DE FIN DE MÁSTER\EDOMAXp.txt'
P_max_txt = open(path_P_max, 'r')
P_max = np.array([float(x.split()[0]) for x in P_max_txt.readlines()])
P_max_txt.close()

path_Rho_max = r'C:\Users\Raul\Desktop\Unversidad de Salamanca\TRABAJO DE FIN DE MÁSTER\EDOMAXro.txt'
Rho_max_txt = open(path_Rho_max, 'r')
Rho_max = np.array([float(x.split()[0]) for x in Rho_max_txt.readlines()])
Rho_max_txt.close()

Rho_max_coef = si.splrep(P_max, Rho_max)


path_P_mid = r'C:\Users\Raul\Desktop\Unversidad de Salamanca\TRABAJO DE FIN DE MÁSTER\EDOMIDp.txt'
P_mid_txt = open(path_P_mid, 'r')
P_mid = np.array([float(x.split()[0]) for x in P_mid_txt.readlines()])
P_mid_txt.close()

path_Rho_mid = r'C:\Users\Raul\Desktop\Unversidad de Salamanca\TRABAJO DE FIN DE MÁSTER\EDOMIDro.txt'
Rho_mid_txt = open(path_Rho_mid, 'r')
Rho_mid = np.array([float(x.split()[0]) for x in Rho_mid_txt.readlines()])
Rho_mid_txt.close()

Rho_mid_coef = si.splrep(P_mid, Rho_mid)


path_P_min = r'C:\Users\Raul\Desktop\Unversidad de Salamanca\TRABAJO DE FIN DE MÁSTER\EDOMINp.txt'
P_min_txt = open(path_P_min, 'r')
P_min = np.array([float(x.split()[0]) for x in P_min_txt.readlines()])
P_min_txt.close()

path_Rho_min = r'C:\Users\Raul\Desktop\Unversidad de Salamanca\TRABAJO DE FIN DE MÁSTER\EDOMINro.txt'
Rho_min_txt = open(path_Rho_min, 'r')
Rho_min = np.array([float(x.split()[0]) for x in Rho_min_txt.readlines()])
Rho_min_txt.close()

Rho_min_coef = si.splrep(P_min, Rho_min)




#definimos los parámetros (constantes)
kappa = 8*np.pi
alpha = -0.05

#Definimos la ecuación de estado
def EoS(p):
    return si.splev(p, Rho_mid_coef)


def fun1(y0, r):
    A, B, B_p, R, R_p, p = y0
    f = R + alpha*R**2
    f_R = 1 + 2*alpha*R
    f_2R = 2*alpha
    f_3R = 0
    rho = EoS(p)
    A_p = 2*r*A/(3*f_R)*(kappa*A*(rho + 3*p) + A*f - f_R*(A*R/2 + 3*B_p/(2*r*B)) - (3/r + 3*B_p/(2*B))*f_2R*R_p)
    B_2p = B_p/2*(A_p/A + B_p/B) + 2*A_p*B/(r*A) + 2*B/f_R*(-kappa*A*p + (B_p/(2*B) + 2/r)*f_2R*R_p - A*f/2)
    R_2p = (R_p*(A_p/(2*A) - B_p/(2*B) - 2/r) - A/(3*f_2R)*(kappa*(rho - 3*p) + f_R*R - 2*f) - f_3R*R_p**2/f_2R)
    p_p = -(rho + p)*B_p/(2*B)
    return [A_p, B_p, B_2p, R_p, R_2p, p_p]



A01 = 1
B01 = 0.21055 #Este valor se obtiene con el shooting method que aparece más adelante (para EoS_mid y p0 = 1e-3: 0.06386 para alpha = -0.05; 0.06387 para alpha = -0.001; 0.0636 para alpha = -0.2 y -0.3)
B_p01 = 0
p01 = 2.4e-4
rho01 = EoS(p01)
T01 = 3*p01-rho01
R01 = -kappa*T01 
R_p01 = 0
y01 = [A01, B01, B_p01, R01, R_p01, p01]

r1 = np.linspace(0,20,1000000)
sol1 = integrate.odeint(fun1, y01, r1[1:])

A1 = np.insert(sol1[:,0], 0, A01)
B1 = np.insert(sol1[:,1], 0, B01)
B_p1 = np.insert(sol1[:,2], 0, B_p01)
R1 = np.insert(sol1[:,3], 0, R01)
R_p1 = np.insert(sol1[:,4], 0, R_p01)
p1 = np.insert(sol1[:,5], 0, p01)
rho1 = EoS(p1)


#Calculamos el valor del radio viendo cuando la presión se hace cero
r_index = list(5e-9<p1).index(False) + 1
rs = r1[r_index-1]
print('El radio de la estrella es de {} km '.format(round(rs,3)))



#recortamos todas las listas hasta el valor del radio, ya que se trata de la solución interior
r1 = r1[:r_index]
A1 = A1[:r_index]
B1 = B1[:r_index]
B_p1 = B_p1[:r_index]
R1 = R1[:r_index]
R_p1 = R_p1[:r_index]
p1 = p1[:r_index]
rho1 = rho1[:r_index]




#calculamos las soluciones numéricas en el exterior
def fun2(y0, r):
    A, B, B_p, R, R_p = y0
    f = R + alpha*R**2
    f_R = 1 + 2*alpha*R
    f_2R = 2*alpha
    f_3R = 0
    A_p = 2*r*A/(3*f_R)*(A*f - f_R*(A*R/2 + 3*B_p/(2*r*B)) - (3/r + 3*B_p/(2*B))*f_2R*R_p)
    B_2p = B_p/2*(A_p/A + B_p/B) + 2*A_p*B/(r*A) + 2*B/f_R*((B_p/(2*B) + 2/r)*f_2R*R_p - A*f/2)
    R_2p = (R_p*(A_p/(2*A) - B_p/(2*B) - 2/r) - A/(3*f_2R)*(f_R*R - 2*f) - f_3R*R_p**2/f_2R)
    return [A_p, B_p, B_2p, R_p, R_2p]


A02 = A1[-1]
B02 = B1[-1]
B_p02 = B_p1[-1]
R02 = R1[-1]
R_p02 = R_p1[-1]
y02 = [A02, B02, B_p02, R02, R_p02]


'''
#Shooting method para calcular B(0)
B_a0 = []
a_list = [10,1000,2000,3000,4000]
for a in a_list:
    r2 = np.linspace(rs,rs*a,10000000)
    sol2 = integrate.odeint(fun2, y02, r2)
    B_a0.append(B1[0]/sol2[:,1][-1])
   

def aux(a,a1,a2,a3):
    return a1 + a2/a**a3

parameters, covariance = curve_fit(aux, a_list, B_a0)
a1, a2, a3 = parameters
fit = aux(a_list, a1, a2, a3)

a_list2 = np.linspace(10,4000,10000)
fit2 = aux(a_list2, a1, a2, a3)

B_0 = aux(1e100, a1, a2, a3) #Este es el valor que hay que poner de B(0) para que B(infty)=1

plt.figure()
plt.plot(a_list, B_a0,'o', label='datos')
plt.plot(a_list2, fit2, label='ajuste')
plt.axhline(B_0, label='B(0)={}'.format(round(B_0,5)), c='k', ls='--')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$B_{\alpha}(0)$')
plt.legend(loc=0)
plt.show()

#calculamos el error del ajuste
e1, e2, e3 = np.sqrt(np.diag(covariance))
error_fit = e1 + a1*e2 + a1*e3 #haciendo propagación de errores de la función aux y considerando a>>1
#A partir de este error sabemos cuantas cifras significativas debemos poner  en B(0)
'''

r2 = np.linspace(rs,20000,10000000)
sol2 = integrate.odeint(fun2, y02, r2)

A2 = sol2[:,0]
B2 = sol2[:,1]
B_p2 = sol2[:,2]
R2 = sol2[:,3]
R_p2 = sol2[:,4]
p2 = np.zeros(len(r2))
rho2 = np.zeros(len(r2))




#combinamos las soluciones interiores y exteriores
r = np.append(r1,r2)
B = np.append(B1,B2)
B_p = np.append(B_p1,B_p2)
A = np.append(A1,A2)
R = np.append(R1,R2)
R_p = np.append(R_p1,R_p2)
p = p1
rho = rho1


M_r = 1/2*(1-B)
M_beta = r*M_r



plt.figure()
plt.plot(r,M_r)
plt.xlim((3500,20000))
plt.ylim((0,0.001))
plt.xlabel('r (km)')
plt.ylabel('M(r)/r')
plt.show()


M_beta_fit = np.polyfit(r[len(r1)+7500000:],M_beta[len(r1)+7500000:],1)
beta = M_beta_fit[0] 
M_beta_val = np.polyval(M_beta_fit, r[len(r1):])

plt.figure()
plt.plot(r[len(r1):],M_beta[len(r1):], label='B(0)={}'.format(B01))
#plt.plot(r[len(r1):], M_beta_val)
plt.legend(loc=0)
plt.xlabel('r (km)')
plt.ylabel(r'$M(\beta,r) (km)$')
plt.ylim((3,3.6))
plt.show()

M = M_beta - beta*r #función M(r) final

U = A*B-1
M_fR = M/(1+U) #función M_fR(r) final

C = (1+2*alpha)**4

plt.figure()
plt.plot(r[len(r1):],M_fR[len(r1):], label=r'$M_{f(R)}(r)$')
plt.plot(r[len(r1):],M[len(r1):], label='M(r)')
plt.axhline(C*M[-1], ls='--')
plt.legend(loc=0)

plt.ylabel(r'$M_{f(R)} (km)$')
plt.ylim((3.5,4.2))
plt.xlim((rs, 20000))
plt.show()

