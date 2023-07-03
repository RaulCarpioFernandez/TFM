# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:13:25 2023

@author: Raul
"""

#Solución de las ecuaciones de campo para f(R) = R + alpha*R^2

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from numpy import diff
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
    return si.splev(p, Rho_max_coef)


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
B01 = 0.161487  #Este valor se obtiene con el shooting method que aparece más adelante (para p0 = 2.4e-4 y alpha=-0.05: 0.2106 para mid, 0.283414 para min y 0.161487 para max)
B_p01 = 0
p01 = 2.4e-4
rho01 = EoS(p01)
T01 = 3*p01-rho01
R01 = -kappa*T01 
R_p01 = 0
y01 = [A01, B01, B_p01, R01, R_p01, p01]

r1 = np.linspace(0,15,100000)
sol1 = integrate.odeint(fun1, y01, r1[1:])

A1 = np.insert(sol1[:,0], 0, A01)
B1 = np.insert(sol1[:,1], 0, B01)
B_p1 = np.insert(sol1[:,2], 0, B_p01)
R1 = np.insert(sol1[:,3], 0, R01)
R_p1 = np.insert(sol1[:,4], 0, R_p01)
p1 = np.insert(sol1[:,5], 0, p01)
rho1 = EoS(p1)


#Calculamos el valor del radio viendo cuando la presión se hace cero
r_index = list(1e-8<p1).index(False) + 1
rs = r1[r_index-1]



#recortamos todas las listas hasta el valor del radio, ya que se trata de la solución interior
r1 = r1[:r_index]
A1 = A1[:r_index]
B1 = B1[:r_index]
B_p1 = B_p1[:r_index]
R1 = R1[:r_index]
R_p1 = R_p1[:r_index]
p1 = p1[:r_index]
rho1 = rho1[:r_index]



#Calculamos la masa de la estrella
M_GR = round(4*np.pi*integrate.cumtrapz(EoS(p1)*r1**2,r1)[-1],3)
print('El radio de la estrella es de {} km y su masa es de {} km'.format(round(rs,3), M_GR))



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
#Sale un error de 0.00005, así que solo tenemos en cuenta 5 cifras significativas en B(0)
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



#Calculamos R de forma alternativa a partir de A y B
B_p = diff(B)/diff(r)
#B_p = np.insert(B_p, 0, B_p[0])
B_2p = diff(B_p)/diff(r[1:])
#B_2p = np.insert(B_2p, 0, B_2p[0])
A_p = diff(A)/diff(r)
#A_p = np.insert(A_p, 0, A_p[0])
R_AB = B_p[1:]/(2*A[2:]*B[2:])*(A_p[1:]/A[2:]+B_p[1:]/B[2:])-B_2p/(A[2:]*B[2:])-2*B_p[1:]/(r[2:]*A[2:]*B[2:])+2*A_p[1:]/(r[2:]*A[2:]**2)-2/(A[2:]*r[2:]**2)+2/r[2:]**2
#R_AB = B_p/(2*A*B)*(A_p/A+B_p/B)-B_2p/(A*B)-2*B_p/(r*A*B)+2*A_p/(r*A**2)-2/(A*r**2)+2/r**2



#Calculamos las soluciones con Schwarzschild para comparar
r_in = np.linspace(0,rs,1000)
A_in = 1/(1-2*M_GR*r_in**2/rs**3)
B_in = 1/4*((3*np.sqrt(1-2*M_GR/rs) - np.sqrt(1-2*M_GR*r_in**2/rs**3))**2)

r_out = np.linspace(rs,300,1000)
A_out = 1/(1-2*M_GR/r_out)
B_out = (1-2*M_GR/r_out)

r_bch = np.append(r_in,r_out)
A_sch = np.append(A_in,A_out)
B_sch = np.append(B_in,B_out)


#representamos las soluciones numéricas
aux_index = list(300>r2).index(False)
b = int(len(r1)+aux_index)


plt.figure()
plt.plot(r[:b],A[:b])
plt.axvline(rs,c='r',ls='dotted',label='$r_b$')
plt.legend(loc=0)
plt.ylabel('A(r)')
plt.xlabel('r (km)')
plt.show()




plt.figure()
plt.plot(r[:b],B[:b])
plt.axvline(rs,c='r',ls='dotted',label='$r_b$')
plt.legend(loc=0)
plt.ylabel('B(r)')
plt.xlabel('r (km)')
plt.show()



plt.figure()
plt.plot(r1,p)
plt.ylabel(r'p ($km^{-2}$)')
plt.xlabel('r (km)')
plt.show()

plt.figure()
plt.plot(r1,rho)
plt.ylabel(r'$\rho (km^{-2})$')
plt.xlabel('r (km)')
plt.show()



plt.figure()
plt.ylim(min(R)*1.2,max(R)*1.2)
plt.plot(r[:int(b*0.9)],R[:int(b*0.9)],label='iteration')
plt.plot(r[10:int(b*0.9)],R_AB[10:int(b*0.9)],label='from A y B',ls='dotted',c='cyan')
plt.axvline(rs,c='r',ls='dotted',label='$r_b$')
plt.legend(loc=0)
plt.ylabel(r'R ($km^{-2}$)')
plt.xlabel('r')
plt.show()


plt.figure()
plt.plot(r[:int(b*0.9)],R_p[:int(b*0.9)])
plt.axvline(rs,c='r',ls='dotted',label='$r_b$')
plt.legend(loc=0)
plt.ylabel('$R_p (r)$')
plt.xlabel('r')
plt.show()

'''
#Metemos las soluciones en las ecuaciones diferenciales para comprobar que los resultados son correctos
s = 1
f = R1 + alpha*R1**2
f_R = 1 + 2*alpha*R1
f_2R = 2*alpha*np.ones(len(R1))
f_3R = np.zeros(len(R1))
Dif_A1 = diff(A1[s-1:])/diff(r1[s-1:]) - 2*r1[s:]*A1[s:]/(3*f_R[s:])*(kappa*A1[s:]*(rho1[s:]+3*p1[s:]) + A1[s:]*f[s:] - f_R[s:]*(A1[s:]*R1[s:]/2 + 3*diff(B1[s-1:])/diff(r1[s-1:])/(2*r1[s:])) - (3/r1[s:] + 3*diff(B1[s-1:])/diff(r1[s-1:])/2)*f_2R[s:]*R_p1[s:])
Dif_B1 = diff(diff(B1)/diff(r1))/diff(r1[1:]) - diff(B1[:-1])/diff(r1[:-1])/2*((diff(A1[:-1])/diff(r1[:-1]))/A1[1:-1]-diff(B1[:-1])/diff(r1[:-1])/B1[1:-1]) - 2*(diff(A1[:-1])/diff(r1[:-1]))*B1[1:-1]/(r1[1:-1]*A1[1:-1]) - 2*B1[1:-1]/f_R[1:-1]*(-kappa*A1[1:-1]*p1[1:-1] + (diff(B1[:-1])/diff(r1[:-1])/2+2/r1[1:-1])*f_2R[1:-1]*R_p1[1:-1] - A1[1:-1]*f[1:-1]/2)
Dif_R1 = diff(diff(R1)/diff(r1))/diff(r1[1:]) - diff(R1[:-1])/diff(r1[:-1])*((diff(A1[:-1])/diff(r1[:-1]))/(2*A1[1:-1]) - (diff(B1[:-1])/diff(r1[:-1]))/(2*B1[1:-1]) - 2/r1[1:-1]) + A1[1:-1]/(3*f_2R[1:-1])*(kappa*(rho1[1:-1] - 3*p1[1:-1]) + f_R[1:-1]*R1[1:-1] - 2*f[1:-1]) + f_3R[1:-1]*(diff(R1[:-1])/diff(r1[:-1]))**2/f_2R[1:-1]
Dif_p1 = diff(p1[s-1:])/diff(r1[s-1:]) + (rho1[s:] + p1[s:])*diff(B1[s-1:])/diff(r1[s-1:])/2



f = R2 + alpha*R2**2
f_R = 1 + 2*alpha*R2
f_2R = 2*alpha*np.ones(len(R2))
f_3R = np.zeros(len(R2))
Dif_A2 = diff(A2)/diff(r2) - 2*r2[1:]*A2[1:]/(3*f_R[1:])*(A2[1:]*f[1:] - f_R[1:]*(A2[1:]*R2[1:]/2 + 3*diff(B2)/diff(r2)/(2*r2[1:])) - (3/r2[1:] + 3*diff(B2)/diff(r2)/2)*f_2R[1:]*R_p2[1:])
Dif_B2 = diff(diff(B2)/diff(r2))/diff(r2[1:]) - diff(B2[1:])/diff(r2[1:])/2*((diff(A2[1:])/diff(r2[1:]))/A2[1:-1]-(diff(B2[1:])/diff(r2[1:]))/B2[1:-1]) - 2*(diff(A2[1:])/diff(r2[1:]))/(r2[1:-1]*A2[1:-1]) - 2/f_R[1:-1]*((diff(B2[1:])/diff(r2[1:])/2+2/r2[1:-1])*f_2R[1:-1]*R_p2[1:-1] - A2[1:-1]*f[1:-1]/2)
Dif_R2 = diff(diff(R2)/diff(r2))/diff(r2[1:]) - diff(R2[1:])/diff(r2[1:])*((diff(A2[1:])/diff(r2[1:]))/(2*A2[1:-1]) - (diff(B2[1:])/diff(r2[1:]))/(2*B2[1:-1]) - 2/r2[1:-1]) + A2[1:-1]/(3*f_2R[1:-1])*(f_R[1:-1]*R2[1:-1] - 2*f[1:-1]) + f_3R[1:-1]*(diff(R2[1:])/diff(r2[1:]))**2/f_2R[1:-1]
Dif_p2 = diff(p2)/diff(r2) + (rho2[1:] + p2[1:])*diff(B2)/diff(r2)/2

Dif_A = np.append(Dif_A1,Dif_A2)
Dif_B = np.append(Dif_B1,Dif_B2)
Dif_p = np.append(Dif_p1,Dif_p2)
Dif_R = np.append(Dif_R1,Dif_R2)



plt.figure()
plt.plot(r[s-1:b],Dif_A[:b],label='error en A')
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(r[s-1:b],Dif_B[:b],label='error en B')
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(r[s-1:b],Dif_p[:b],label='error en p')
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(r[s-1:b],Dif_R[:b],label='error en R')
plt.legend(loc=0)
plt.show()

'''