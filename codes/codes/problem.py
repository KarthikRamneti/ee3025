import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#read .wav file 
input_signal,fs = sf.read('Sound_Noise.wav') 

#sampling frequency of Input signal
sampl_freq=fs

#order of the filter
order=4   

#cutoff frquency 4kHz
cutoff_freq=4000.0  

#digital frequency
Wn=2*cutoff_freq/sampl_freq  

# b and a are numerator and denominator polynomials respectively
b, a = signal.butter(order,Wn, 'low') 

#filter the input signal with butterworth filter

#given method
output_signal = signal.filtfilt(b, a, input_signal)

#own routine
x = input_signal
N = len(x)
k = np.arange(len(x))
w = 2*np.pi*k/len(x)
z = np.exp(1j * w)

#H(z) 
num = np.polyval(b,z**(-1))
den = np.polyval(a,z**(-1))
H = num/den

#X(z)
"""
#own dft but it is O(n2) so it takes long time to compute so used a built in fft function 
N = len(x)
X = np.zeros(N) + 1j*np.zeros(N)
for k in range(0,N):
	for n in range(0,N):
		X[k]+=x[n]*np.exp(-1j*2*np.pi*n*k/N)
"""
X = np.fft.fft(x)

#Y(z) = H(z)X(z)
Y = np.zeros(N) + 1j*np.zeros(N)
for k in range(0,N):
	Y[k] = X[k]*H[k]

#DTFT
"""
#own idft but it is O(n2) so it takes long time to compute so used a built in ifft function 
y = np.zeros(N) + 1j*np.zeros(N)
for k in range(0,N):
	for n in range(0,N):
		y[k]+=Y[n]*np.exp(1j*2*np.pi*n*k/N)

y = np.real(y)/N
"""
y = np.fft.ifft(Y).real

#output_signal = signal.lfilter(b, a, input_signal)

#write the output signal into .wav file
sf.write('Sound_With_ReducedNoise.wav', output_signal, fs) 
sf.write('Sound_With_ReducedNoise_ownroutine.wav', y, fs) 

#Plotting

plt.figure(0)
plt.figure(figsize=(8,7))
plt.subplot(2,1,1)
plt.plot(y,'c')
plt.title('Output with own routine')
plt.grid()

plt.subplot(2,1,2)
plt.plot(output_signal,'b')
plt.title('Output with built in command')
plt.grid()

plt.suptitle("Time Domain Responce")

plt.savefig('../figs/ee18btech11037_1.eps')

plt.figure(1)
plt.figure(figsize=(8,7))
plt.subplot(2,1,1)
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(y))),'c')
plt.title('Output with own routine')
plt.grid()

plt.subplot(2,1,2)
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(output_signal))),'b')
plt.title('Output with built in command')
plt.grid()

plt.suptitle("Frequency Domain Responce")
plt.savefig('../figs/ee18btech11037_2.eps')

#plt.show()

#If using termux
#subprocess.run(shlex.split("termux-open ../figs/ee18btech11037_1.pdf"))
#subprocess.run(shlex.split("termux-open ../figs/ee18btech11037_2.pdf"))
