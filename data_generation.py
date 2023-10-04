
import numpy as np
from scipy.signal import tukey


# warnings.filterwarnings("ignore")

def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data,(0,int((2**pow_2)-N)),'constant')

def FFT(signal_t):
    signal_t_pad = zero_pad(signal_t)
    fft_sig = np.fft.fft(signal_t_pad)
    fft_sig_shift = np.fft.fftshift(fft_sig)
    
    return fft_sig_shift

def h(a,t):
    """
    This is a function. It takes in a value of the amplitude $a$ and a time vector $t$ and spits out whatever
    is in the return function. Modify amplitude to improve SNR. Modify frequency range to also affect SNR but 
    also to see if frequencies of the signal are important for the windowing method. We aim to estimate the parameter
     --a-- .
    """
    f = 1e-5 *t**(1/2)  # "chirping" sinusoid
    
    return 3e-20*(a * np.sin((2*np.pi)*f*t))

def gap_function(t,start_window,end_window,lobe_length, gap_flag):
    if gap_flag == True:

        one_hour = 60*60
        start_window *= one_hour          # Define start of gap
        end_window *= one_hour          # Define end of gap
        lobe_length *= one_hour          # Define length of cosine lobes

        delta_t = t[1] - t[0]
        window_length = int(np.ceil(((end_window+lobe_length) - 
                                    (start_window - lobe_length))/delta_t))  # Construct of length of window 
                                                                            # throughout the gap
        alpha_gaps = 2*lobe_length/(delta_t*window_length)      # Construct alpha (windowing parameter)
                                                        # so that we window BEFORE the gap takes place.
            
        window = tukey(window_length,alpha_gaps)   # Construct window

        gap = []  # Initialise with empty vector
        j=0  
        for i in range(0,len(t)):   # loop index i through length of t
            if t[i] > (start_window - lobe_length) and (t[i] < end_window + lobe_length):  # if t within gap segment
                gap.append(1 - window[j])  # add windowing function to vector of ones.
                j+=1  # incremement 
            else:                   # if t not within the gap segment
                gap.append(1)  # Just add a onne.
                j=0
                

        alpha_full = 0.2
        total_window = tukey(len(gap), alpha = alpha_full)

        gap *= total_window
    else:
        gap = np.ones(len(t))
    return gap


def ts_waveform(params):
    
    waveform_prop = h(params[0],t)
    waveform_prop_pad = zero_pad(waveform_prop)
    waveform_prop_gap = gap_window * waveform_prop_pad

    return waveform_prop_gap


def lprior(params):
    if params[0] < a_low or params[0] > a_high:
        log_prior = -np.inf
    else:
        log_prior = 0
    return log_prior
def lpost(params):
    '''
    Compute log posterior
    '''
    if np.isinf(lprior(params)):
        return -np.inf  # Don't even bother calculating the likelihood if we fall out of the prior range. No support. 
    else:
        return(lprior(params) + llike(params))
def llike(params):
    
    ts_waveform(params)

    waveform_prop_gap_fft = FFT(waveform_prop_gap)

    diff_f_gap = data_f_gap - waveform_prop_gap_fft
    return(-0.5*np.real(diff_f_gap.conj() @ Cov_Matrix_gap_inv @ diff_f_gap))