from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import pickle



def Equilibrium_Reconstruct(defl_data,k,T,window_crit=[0,None],windowsize=500,bandwidth=10e-11,bin_number=100,portion='Apr',defl_offset=0,detail='n'):
    '''Find the window of point of equilibrium and presented energy landscape reconstruction (over a defelction coordinate) from deflection data of an AFM force-distance measurement
    -----Output-----
    Default output is a dataframe with columns: deflection, reconstructed energy 
    More detail can be output if type in a string with following characters (case sensitive):  
	'E': Measured combined energy (adding to the dataframe)
	'V': Probe potentail energy (adding to the dataframe)
	'P': Probabolity density function of the probe (adding to the dataframe)
	'd': Representative bond/unbond deflection position at point of equilibrium (adding as new element in the list)
	'F': Representative binding/unbinding force at point of equilibrium (adding as new element in the list)
	'r': raw deflection data of the window of point of equilibrium (adding as new element in the list)
    -----Parameters-----
    defl_data: Deflection data of a force-distance measurement
    k: spring constant of the force probe
    T: equivalent temperature of the probe
    window_crit: imput as a list of two numbers if only want to consider a specific critical window of the input dataset
    windowsize: define the size of the scanning window  
    bandwidth: define the bandwidth to implement kernel density estimation (KDE)
    bin_number: number of bins for histogram plot for KDE
    portion: indicate if input dataset is a approach ('Apr', 'apr', or 'APR') or retract ('Ret','ret','RET') portion of the force-distance measurement 
    defl_offset: offset the deflection data if the offset of deflection data is known 
    detail: type in the corresponding characters if wanting addional information 
    -----Returns-----    '''
	k_B=1.380649e-23

	kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
	if portion=='Apr' or portion=='apr' or portion=='APR':
		data=np.asarray(defl_data)[window_crit[0]:window_crit[1]]
	elif portion=='Ret' or portion=='ret' or portion=='RET':
		data=np.flip(np.asarray(defl_data))[window_crit[0]:window_crit[1]]
	else:
		print('Wrong input for portion')
	step_ticks=10
	scan_steps=int(windowsize/step_ticks)
	number_steps=int(len(data)/scan_steps)
	bin_number=bin_number#int(windowsize/50)
	#min_half_probability=1
	#Calculate offset with first window first
	for i in [0]:
		this_window=data[i*scan_steps:(i*scan_steps+windowsize)]
		sample_range=np.linspace(np.min(this_window),np.max(this_window),bin_number)
		kde.fit(np.asarray(this_window[:, None]))
		logprob = kde.score_samples(sample_range[:, None])
		pdf = np.cumsum(np.exp(logprob)*(np.ptp(this_window)/(bin_number-1)))
		half_index=np.abs(pdf-0.5).argmin()
		this_half_probability=(np.exp(logprob)/(bin_number-1))[half_index]
		min_half_probability=this_half_probability
		offset=np.mean(this_window)-(defl_offset)
	#test
	#print('check 1')
	#print('P_half_min=',min_half_probability)
	for i in range(number_steps-step_ticks-1):
		i+=1
		this_window=data[i*scan_steps:(i*scan_steps+windowsize)]
		sample_range=np.linspace(np.min(this_window),np.max(this_window),bin_number)
		kde.fit(np.asarray(this_window[:, None]))
		logprob = kde.score_samples(sample_range[:, None])
		pdf = np.cumsum(np.exp(logprob)*(np.ptp(this_window)/(bin_number-1)))##try .values
		half_index=np.abs(pdf-0.5).argmin()
		this_half_probability=(np.exp(logprob)/(bin_number-1))[half_index]
		#test
		#print('check 2')
		#print('P_This_half',this_half_probability)
		#i_test=i
		if this_half_probability<min_half_probability:
			min_half_probability=this_half_probability
			window_eq=np.exp(logprob)*(np.ptp(this_window/(bin_number-1)))##try .values
			raw_window=this_window-offset
			sample_range_eq=sample_range
			half_point=sample_range[half_index]
			defl_eq_b=sample_range[np.exp(logprob)[:half_index].argmax()]-np.median(data[:(5*windowsize)])
			defl_eq_u=sample_range[np.exp(logprob)[half_index:].argmax()+half_index]-np.median(data[:(5*windowsize)])
			window_index=np.arange(windowsize)+(i*scan_steps)
			#print('check_3')
	#test
	#print('check 3')
	inter_eng=np.log(window_eq/np.max(window_eq))*-k_B*T
	probe_eng=0.5*k*(sample_range_eq-offset)**2
	Recon_eng=pd.DataFrame(dict(defl=sample_range_eq-offset,Energy=inter_eng-probe_eng))

		
	if 'P' or 'V' or 'E' or 'F' or 'r' or 'd' in detail:
		list_return=[]
		if 'E' in detail:
			Recon_eng=pd.concat([Recon_eng,pd.DataFrame(dict(E=inter_eng))],axis=1)
			#list_return.append(inter_eng)
		if 'V' in detail:
			Recon_eng=pd.concat([Recon_eng,pd.DataFrame(dict(V=probe_eng))],axis=1)
			#list_return.append(probe_eng)
		if 'P' in detail:
			Recon_eng=pd.concat([Recon_eng,pd.DataFrame(dict(P=window_eq/np.max(window_eq)))],axis=1)
			#list_return.append(window_eq/np.max(window_eq))
		list_return.append(Recon_eng)
		if 'd' in detail:
			list_return.append(pd.DataFrame(dict(d_u=[defl_eq_u],d_b=[defl_eq_b])))
		if 'F' in detail:
			list_return.append(pd.DataFrame(dict(F_u=[defl_eq_u*k],F_b=[defl_eq_b*k])))
		if 'r' in detail:
			list_return.append(pd.DataFrame(dict(index=window_index,defl=raw_window)))
		return(list_return)
	else:
		return(Recon_eng)

