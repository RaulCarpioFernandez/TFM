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

#Definimos la ecuación de estado
def EoS(p):
    return si.splev(p, Rho_mid_coef)



def funGR(y0, r):
    A, psi, p = y0
    rho = EoS(p)
    R = kappa*(rho - 3*p)
    A_p = 2*r*A/3*(kappa*A*(rho+3*p) + A*R/2 - 3*psi/(2*r))
    psi_p = psi/2*(A_p/A-psi) + 2*A_p/(r*A) - 2*kappa*A*p - A*R
    p_p = -(rho + p)*psi/2
    return [A_p, psi_p, p_p]
R_totalGR = []
M_totalGR= []

r_aux = np.linspace(12,15,100)


p_list = np.linspace(1e-4,1.5e-3,20)
for pi in p_list:
    A01 = 1
    psi01 = 0
    p01 = pi
    rho01 = EoS(p01)
    T01 = 3*p01-rho01
    R01 = -kappa*T01 
    R_p01 = 0
    y01 = [A01, psi01, p01]
    
    r1 = np.linspace(0,20,10000)
    sol1 = integrate.odeint(funGR, y01, r1[1:])
    p1 = np.insert(sol1[:,2], 0, p01)
    rho1 = EoS(p1)
    
    
    #Calculamos el valor del radio viendo cuando la presión se hace cero y con ese 
    #valor calculamos la masa de la estrella
    r_index = list(5e-7<p1).index(False) 
    R_totalGR.append(r1[r_index-1])
    r1 = r1[:r_index]
    p1 = p1[:r_index]
    
    #Calculamos la masa a partir del radio de la estrella
    M_totalGR.append(round(0.67818*4*np.pi*integrate.cumtrapz(EoS(p1)*r1**2,r1)[-1],3))
    


plt.figure()
alpha_list = [-0.001,-0.05]
for alpha in alpha_list:    
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
    
    
    M_total = []
    R_total = []
    p_list = np.linspace(1e-4,1.5e-3,9)
    for pi in p_list:
        A01 = 1
        B01 = 1
        B_p01 = 0
        p01 = pi
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
        A02 = A1[-1]
        B02 = B1[-1]
        B_p02 = B_p1[-1]
        R02 = R1[-1]
        R_p02 = R_p1[-1]
        y02 = [A02, B02, B_p02, R02, R_p02]
        
        
        
        #Shooting method para calcular B(0)
        B_a0 = []
        a_list = [10,1000,2000,3000,4000]
        for a in a_list:
            r2b = np.linspace(rs,rs*a,10000000)
            sol2 = integrate.odeint(fun2, y02, r2b)
            B_a0.append(B1[0]/sol2[:,1][-1])
           
        
        def aux(a,a1,a2,a3):
            return a1 + a2/a**a3
        
        parameters, covariance = curve_fit(aux, a_list, B_a0)
        a1, a2, a3 = parameters
        fit = aux(a_list, a1, a2, a3)
        
        a_list2 = np.linspace(10,4000,10000)
        fit2 = aux(a_list2, a1, a2, a3)
        
        B_0 = aux(1e100, a1, a2, a3) #Este es el valor que hay que poner de B(0) para que B(infty)=1
        B01 = B_0 
        y01 = [A01, B01, B_p01, R01, R_p01, p01]
        
        r1 = np.linspace(0,rs,1000000)
        sol1 = integrate.odeint(fun1, y01, r1[1:])
        
        A1 = np.insert(sol1[:,0], 0, A01)
        B1 = np.insert(sol1[:,1], 0, B01)
        B_p1 = np.insert(sol1[:,2], 0, B_p01)
        R1 = np.insert(sol1[:,3], 0, R01)
        R_p1 = np.insert(sol1[:,4], 0, R_p01)
        p1 = np.insert(sol1[:,5], 0, p01)
        rho1 = EoS(p1)    
        
        
        #calculamos las soluciones numéricas en el exterior
        A02 = A1[-1]
        B02 = B1[-1]
        B_p02 = B_p1[-1]
        R02 = R1[-1]
        R_p02 = R_p1[-1]
        y02 = [A02, B02, B_p02, R02, R_p02]
        
        
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
        M_beta_fit = np.polyfit(r[len(r1)+7500000:],M_beta[len(r1)+7500000:],1)
        beta = M_beta_fit[0] 
        M_beta_val = np.polyval(M_beta_fit, r[len(r1):])
        M = M_beta - beta*r #función M(r) final
        
        M_total.append(M[-1]*0.67818)
        R_total.append(rs)
    
    plt.scatter(R_total, M_total, label=r'$\alpha = {}$'.format(alpha))
    plt.plot(R_total, M_total)
plt.plot(R_totalGR, M_totalGR, label='GR')    
plt.legend(loc=0)
plt.show()

