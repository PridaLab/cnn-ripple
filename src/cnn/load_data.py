import scipy.io
import pandas as pd
import numpy as np
import h5py
import sys
import os

import bz_LoadBinary


def load_info (path):
	try:
		mat = scipy.io.loadmat(path+"/info.mat")
	except:
		print("info.mat file does not exist.")
		sys.exit()

	fs = mat["fs"][0][0]
	expName = mat["expName"][0]

	ref_channels = {}
	ref_channels["so"] = mat["so"][0]
	ref_channels["pyr"] = mat["pyr"][0]
	ref_channels["rad"] = mat["rad"][0]
	ref_channels["slm"] = mat["slm"][0]


	if len(mat["chDead"]) <= 0:
		dead_channels = []
	else:
		dead_channels = [x-1 for x in (mat["chDead"][0]).astype(int)]

	return fs, expName, ref_channels, dead_channels

def load_ripples (path, verbose=False):
	try:
		dataset = pd.read_csv(path+"/ripples.csv", delimiter=' ', header=0, usecols = ["ripIni", "ripMiddle", "ripEnd", "type", "shank"])
	except:
		print(path+"/ripples.csv file does not exist.")
		sys.exit()

	ripples = dataset.values
	ripples = ripples[np.argsort(ripples, axis=0)[:, 0], :]
	if verbose:
		print("Loaded ripples: ", len(ripples))

	return ripples

def load_channels_map (path):
	try:
		dataset = pd.read_csv(path+"/mapsCh.csv", delimiter=' ', header=0)
	except:
		print("ripples.csv file does not exist.")
		sys.exit()

	channels_map = dataset.values

	return channels_map

def reformat_channels (channels_map, ref_channels):
	channels = np.where(np.isnan(channels_map[:, 0]) == False, channels_map[:, 0], 0)
	channels = [x-1 for x in (channels).astype(int)]

	shanks = np.where(np.isnan(channels_map[:, 1]) == False, channels_map[:, 1], 0)
	shanks = [x-1 for x in (shanks).astype(int)]

	ref_channels["so"] = np.where(np.isnan(ref_channels["so"]) == False, ref_channels["so"], 0)
	ref_channels["so"] = [x-1 for x in ref_channels["so"].astype(int)]
	ref_channels["pyr"] = np.where(np.isnan(ref_channels["pyr"]) == False, ref_channels["pyr"], 0)
	ref_channels["pyr"] = [x-1 for x in ref_channels["pyr"].astype(int)]
	ref_channels["rad"] = np.where(np.isnan(ref_channels["rad"]) == False, ref_channels["rad"], 0)
	ref_channels["rad"] = [x-1 for x in ref_channels["rad"].astype(int)]
	ref_channels["slm"] = np.where(np.isnan(ref_channels["slm"]) == False, ref_channels["slm"], 0)
	ref_channels["slm"] = [x-1 for x in ref_channels["slm"].astype(int)]

	return channels, shanks, ref_channels

def load_raw_data (path, expName, channels, verbose=False):
	
	# There is .dat file
	is_dat = any([file.endswith(".dat") for file in os.listdir(path)])

	# There is .eeg file
	is_eeg = any([file.endswith(".eeg") for file in os.listdir(path)])
	
	# There is .mat file with the name of the last folder
	is_mat = any([os.path.basename(os.path.normpath(path))+".mat" in file for file in os.listdir(path)])

	if is_dat:
		name_dat = os.listdir(path)[np.where([file.endswith(".dat") for file in os.listdir(path)])[0][0]]
		if verbose:
			print(path+"/"+name_dat)
		data = bz_LoadBinary.bz_LoadBinary(path+"/"+name_dat, len(channels), channels, 2, verbose)

	elif is_eeg:
		name_eeg = os.listdir(path)[np.where([file.endswith(".eeg") for file in os.listdir(path)])[0][0]]
		if verbose:
			print(path+"/"+name_eeg)
		data = bz_LoadBinary.bz_LoadBinary(path+"/"+name_eeg, len(channels), channels, 2, verbose)

	elif is_mat:
		folder = path + "/" + os.path.basename(os.path.normpath(path))+".mat"
		if verbose:
			print(folder)
		try:
			mat = scipy.io.loadmat(folder)
			data = mat["fil"]
		except:
			mat = h5py.File(folder, 'r')
			data = np.array(mat["fil"]).T
	else:
		print('Not data found')

	return data


def downsample_data (data, fs, downsampled_fs):

    # Dowsampling
    if fs > downsampled_fs:
        downsampled_pts = np.linspace(0, data.shape[0]-1, int(np.round(data.shape[0]/fs*downsampled_fs))).astype(int)
        downsampled_data = data[downsampled_pts, :]

    # Upsampling
    elif fs < downsampled_fs:
        print("Original sampling rate below 1250 Hz!")
        return None


    # Change from int16 to float16 if necessary
    # int16 ranges from -32,768 to 32,767
    # float16 has ±65,504, with precision up to 0.0000000596046
    if downsampled_data.dtype != 'float16':
        downsampled_data = np.array(downsampled_data, dtype="float16")

    return downsampled_data


def z_score_normalization(data):
	channels = range(np.shape(data)[1])

	for channel in channels:
		# Since data is in float16 type, we make it smaller to avoid overflows
		# and then we restore it.
		# Mean and std use float64 to have enough space
		# Then we convert the data back to float16
		dmax = np.amax(data[:, channel])
		dmin = abs(np.amin(data[:, channel]))
		dabs = dmax if dmax>dmin else dmin
		m = np.mean(data[:, channel] / dmax, dtype='float64') * dmax
		s = np.std(data[:, channel] / dmax, dtype='float64') * dmax
		s = 1 if s == 0 else s # If std == 0, change it to 1, so data-mean = 0
		data[:, channel] = ((data[:, channel] - m) / s).astype('float16')
	
	return data


def load_data(path, shank, verbose=False):
	# Read info.mat
	fs, expName, ref_channels, dead_channels = load_info(path)

	# Read ripples.csv
	ripples = load_ripples(path, verbose)

	#Read mapsCh.csv
	channels_map = load_channels_map(path)

	# Reformat channels into correct values
	channels, shanks, ref_channels = reformat_channels(channels_map, ref_channels)

	# Read .dat
	data = load_raw_data(path, expName, channels, verbose=verbose)

	# Select channels to make dataset
	channels_in_shank = list(np.where(np.array(shanks)==shank-1)[0])
	data = data[:, channels_in_shank]

	return data, fs


def generate_overlapping_windows(data, window_size, stride, fs):
	window_pts = int(window_size * fs)
	stride_pts = int(stride * fs)
	r = range(0, data.shape[0], stride_pts)

	new_data = np.empty((len(list(r)), window_pts, data.shape[1]))

	cont = 0
	for idx in r:
		win = data[idx:idx+window_pts, :]

		if (win.shape[0] < window_pts):
			continue

		new_data[cont,:,:]  = win

		cont = cont+1

	return new_data
