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
