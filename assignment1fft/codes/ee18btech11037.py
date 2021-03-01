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

#fft function using Cooley–Tukey FFT algorithm
def fft(x):
	# Making the length of x as the power of 2 by padding zeroes if necessary
	N = len(x)
	if np.log2(N) == int(np.log2(N)):
		return fft_2(x)
	else:
		n = int(np.log2(N)) +1
		N_new = 2**n
		x=np.pad(x, (0,N_new-N), 'constant', constant_values=(0))
		return fft_2(x)

def fft_2(x): #recursive algorithm used in Cooley–Tukey FFT algorithm with time complexity O(nlogn)
	N = len(x)
	if N==1:
		return x
	x_even = x[::2]
	x_odd = x[1::2]
	X_even = fft(x_even)
	X_odd = fft(x_odd)
	w = np.exp(-2*np.pi*(1j)/N)
	X = [0+0j]*N
	for k in range(N//2):
		X[k] = X_even[k] + (w**k)*X_odd[k]
		X[k+N//2] = X_even[k] - (w**k)*X_odd[k]
	return X

#ifft function using Cooley–Tukey IFFT algorithm
def ifft(x): #recursive algorithm used in Cooley–Tukey IFFT algorithm with time complexity O(nlogn)
	N = len(x)
	if N==1:
		return x
	x_even = x[::2]
	x_odd = x[1::2]
	X_even = ifft(x_even)
	X_odd = ifft(x_odd)
	w = np.exp(2*np.pi*(1j)/N)
	X = [0+0j]*N
	for k in range(N//2):
		X[k] = (X_even[k] + (w**k)*X_odd[k])/2
		X[k+N//2] = (X_even[k] - (w**k)*X_odd[k])/2
	return X

#X(z)
X = fft(x)

k = np.arange(len(X))
w = 2*np.pi*k/len(X)
z = np.exp(1j * w)

#H(z) 
num = np.polyval(b,z**(-1))
den = np.polyval(a,z**(-1))
H = num/den

#Y(z) = H(z)X(z)
Y = np.zeros(len(X)) + 1j*np.zeros(len(X))
for k in range(0,len(X)):
	Y[k] = X[k]*H[k]

#DTFT
y = np.real(ifft(Y))
y = y[:N]


#write the output signals by both the methods into .wav file
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

plt.suptitle("Time Responce")

#plt.savefig('../figs/ee18btech11037_1.eps')

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

plt.suptitle("Frequency Responce")
#plt.savefig('../figs/ee18btech11037_2.eps')

plt.show()

#If using termux
#subprocess.run(shlex.split("termux-open ../figs/ee18btech11037_1.pdf"))
#subprocess.run(shlex.split("termux-open ../figs/ee18btech11037_2.pdf"))
