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

n = int(len(input_signal))
n = int(2 ** np.floor(np.log2(n)))
input_signal = input_signal[0 : n]

#Passing butterworth filter
b, a = signal.butter(order, Wn, 'low')

# Computing H(z)
w = 2 * np.pi * np.arange(n)/n

#filter the input signal with butterworth filter

def to_complex(field):
	field = str(field)[2:]
	field = field[0 : len(field) - 1]
	return complex(field.replace('+-', '-').replace('i', 'j'))

#output using a c program
Y1 = np.loadtxt('Y.dat', converters={0: to_complex}, dtype = np.complex128, delimiter = '\n') # Load the DFT computed using fft.c
y = np.loadtxt('y.dat', converters={0: to_complex}, dtype = np.complex128, delimiter = '\n').real

#output usig inbuilt filter function
#given method
output_signal = signal.filtfilt(b, a, input_signal)


#write the output signals by both the methods into .wav file
sf.write('Sound_With_ReducedNoise.wav', output_signal, fs) 
sf.write('Sound_With_ReducedNoise_ownroutine.wav', y, fs) 


#Plotting
w[0] = -np.pi
for i in range(1, n):
	w[i] = w[i - 1] + 2 * np.pi/n

# Time domain
t = np.arange(0, n / sampl_freq, 1/sampl_freq)
f2 = plt.figure(figsize = (10, 10))
plt.subplot(2, 1, 1)
plt.plot(t, y,'c')
plt.xlabel("t (in sec)")
plt.ylabel("y(t)")
plt.grid()
plt.title("Output using a c program")

plt.subplot(2, 1, 2)
plt.plot(t, output_signal,'b')
plt.xlabel("t (in sec)")
plt.ylabel("y(t)")
plt.title("Output with built in command")
plt.grid()
plt.suptitle("Time Responce")
#plt.savefig('../figs/ee18btech11037_1.eps')

# Frequency domain
f1 = plt.figure(figsize = (10, 10))
plt.subplot(2, 1, 1)
plt.plot(w, abs(Y1),'c')
plt.xlabel("w (in rad)")
plt.ylabel("|Y(w)|")
plt.grid()
plt.title("Output using a c program")

plt.subplot(2, 1, 2)
plt.plot(w, abs(np.fft.fftshift(np.fft.fft(output_signal))),'b')
plt.xlabel("w (in rad)")
plt.ylabel("|Y(w)|")
plt.grid()
plt.title("Output with built in command")
plt.suptitle("Frequency Responce")
#plt.savefig('../figs/ee18btech11037_2.eps')
plt.show()




#If using termux
#subprocess.run(shlex.split("termux-open ../figs/ee18btech11037_1.pdf"))
#subprocess.run(shlex.split("termux-open ../figs/ee18btech11037_2.pdf"))



