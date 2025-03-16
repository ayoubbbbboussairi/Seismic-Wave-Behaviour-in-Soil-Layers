
import numpy as np
import matplotlib.pyplot as plt

#Constants:
theta=np.array([np.radians(0), np.radians(0), np.radians(0), np.radians(0), np.radians(0), np.radians(0), np.radians(0)], dtype=np.float64) #Degrees

v=np.array([0.33, 0.33, 0.33, 0.48, 0.47, 0.49, 0.19], dtype=np.float64) 
rho=np.array([1700, 1800, 1800, 2000, 2200, 2300, 2600], dtype=np.float64) #Mass density [kg/m^3]
E=np.array([180, 300, 300, 530, 1200, 3300, 4200], dtype=np.float64) #Young Modulus [MPa]
h=np.array([50, 37, 10, -20, -63, -130, -145], dtype=np.float64) #Layer height [m]
tau=40 #Relaxation time [s]
#V_s=np.array([1000, 600, 400], dtype=np.float64)

f=np.arange(0.01, 14, 0.001)


#Imaginary number
j=complex(0,1)

#Chosen values:
A_1=1
#Q=0.01 #Attenuation factor: value between 0.01-0.05

A_i=np.empty((7, len(f)), dtype=np.complex128)
A_dashi=np.empty((7, len(f)), dtype=np.complex128)
A_i[0]=A_1
A_dashi[0]=A_1
#A_i=np.append(A_i, A_1)
#A_dashi=np.append(A_i, A_1)

#Group number:
n=2

#Empty vectors for properties:
X_i=np.empty(6, dtype=np.float64)
k_zi=np.empty((6, len(f)), dtype=np.float64)
Q=np.empty((6, len(f)), dtype=np.float64)
f_i=np.empty((7, len(f)), dtype=np.complex128)
g_i=np.empty((7, len(f)), dtype=np.complex128)
T_locali=np.empty((6, len(f)), dtype=np.complex128)
T_i=np.empty((1, len(f)), dtype=np.complex128)
#T=np.empty((3, len(f)))

#Counter
i=0
k=0

#Distance:
x=2000+600*(n-1) #m

#Layer thickness
h_layer=np.array([], dtype=np.float64)

for i in range(6):
    h_layer_i=h[i]-h[i+1] #m
    h_layer=np.append(h_layer, h_layer_i) #m
    i=i+1
i=0
print(h_layer)

#Angular frequency 
w=2*np.pi*f

#Properties calculation:    
#Shear modulus:
mu_i=E/(2*(1+v)) #MPa
print(mu_i)
#print(f"Case {i + 1}: mu_i={mu_i}")
#print(f"rho[i]={rho[i]}, V_s[i]={V_s[i]}")

#Layer velocity:
V_s=np.sqrt(mu_i*10**6/rho) #m/s
print(V_s)

#Dynamic viscosity:
eta=mu_i*tau*10**6
print(eta)

#Attenuation factor:
E_v=[[float(x)] * len(f) for x in E]
eta_v=[[float(x)] * len(f) for x in eta]
#'''
#print(E_v[0])
for k in range(6): #Maxwell
    Q_k=E_v[k]*np.array([10**6])/(w*eta_v[k])
    Q[k]=1/Q_k
    k=k+1
k=0
#'''
'''
for k in range(6): #Kelvin
    Q_k=(w*eta_v[k])/(np.array([10**6])*E_v[k])
    Q[k]=Q_k
    k=k+1
k=0
print(Q)
'''
#Betha:
Betha=Q/(2)

#Strange X factor:
for k in range(6):
    X_k=np.sqrt(mu_i[k]*rho[k]/(mu_i[k+1]*rho[k+1]))*np.cos(theta[k])/np.cos(theta[k+1])
    X_i[k]=X_k
    #print(f"X_i={X_k}")
    k=k+1
k=0
#print(X_i)
    
#Vertical component of the wave number (real part):
for k in range(6):
    k_zk=w*np.cos(theta[k])/V_s[k]   
    k_zi[k]=k_zk
    #print(k_zi)
    k=k+1
k=0
print(k_zi[i])    
#Vertical component of the wave number (imaginary part):
k_stari=k_zi*(1+j*Betha)
#print(k_stari)

#A factors of the transfer functions:
for k in range(1,6):
    A_k=(1/2)*A_i[k-1]*(1+X_i[k-1])*np.exp(-j*k_stari[k-1]*h_layer[k-1])+(1/2)*A_dashi[k-1]*(1-X_i[k-1])*np.exp(j*k_stari[k-1]*h_layer[k-1])
    A_i[k]=A_k 
    A_dashk=(1/2)*A_i[k-1]*(1-X_i[k-1])*np.exp(-j*k_stari[k-1]*h_layer[k-1])+(1/2)*A_dashi[k-1]*(1+X_i[k-1])*np.exp(j*k_stari[k-1]*h_layer[k-1])
    A_dashi[k]=A_dashk    
    k=k+1
k=6
A_k=(1/2)*A_i[k-1]*(1+X_i[k-1])*np.exp(-j*k_stari[k-1]*h_layer[k-1])+(1/2)*A_dashi[k-1]*(1-X_i[k-1])*np.exp(j*k_stari[k-1]*h_layer[k-1])
A_i[6]=A_k 
A_dashi[6]=A_k  
k=0  
    
print(A_dashi[6])
'''
#Functions f and g for transfer function:
for k in range(6):
    f_k=A_i[k]/A_i[0]
    g_k=A_dashi[k]/A_dashi[0]
    f_i[k]=f_k
    g_i[k]=g_k
    k=k+1
k=0
#print(f_i[5])
print(g_i[6])
'''   
AA=A_i[0]
#Local transfer functions:
for k in range(6):
    T_localk=(A_i[k]/AA+A_dashi[k]/AA)/(A_i[k+1]/AA+A_dashi[k+1]/AA)
    T_locali[k]=T_localk
    k=k+1
k=0
print(T_locali[0])

#Global transfer function
T_i=T_locali[0]*T_locali[1]*T_locali[2]*T_locali[3]*T_locali[4]*T_locali[5]
T_1=np.abs(T_i)


plt.plot(f, T_1, label=f'Transfer function (Kelvin-Voigt)')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend()
plt.show()


plt.plot(f, Q[1], label=f'Attenuation factor (Kelvin-Voigt)')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Attenuation factor')
plt.xlim(0,0.2)
plt.ylim(0,10)
plt.legend()
plt.show()

