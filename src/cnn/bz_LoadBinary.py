import numpy as np
import math


def loadChunk(fid, nChannels, channels, nSamples, precision):
	size = int(nChannels * nSamples * precision)
	nSamples = int(nSamples)

	data = fid.read(size)

	# fromstring to read the data as int16
	# reshape to give it the appropiate shape (nSamples x nChannels)
	data = np.fromstring(data, dtype=np.int16).reshape(nSamples, len(channels))
	data = data[:, channels]

	return data



def bz_LoadBinary(filename, nChannels, channels, sampleSize, verbose=False):

	if (len(channels) > nChannels):
		print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
		return

	#aqui iria CdE de filename
	with open(filename, "rb") as f:
		dataOffset = 0

		# Determine total number of samples in file
		fileStart = f.tell()
		if verbose:
			print("fileStart ", fileStart)
		status = f.seek(0, 2) # Go to the end of the file
		fileStop = f.tell()
		f.seek(0, 0) # Back to the begining
		if verbose:
			print("fileStop ", fileStop)

		# (floor in case all channels do not have the same number of samples)
		maxNSamplesPerChannel = math.floor(((fileStop-fileStart)/nChannels/sampleSize))
		nSamplesPerChannel = maxNSamplesPerChannel

		# For large amounts of data, read chunk by chunk
		maxSamplesPerChunk = 10000
		nSamples = int(nSamplesPerChannel*nChannels)

		if verbose:
			print("nSamples ", nSamples)

		if nSamples <= maxNSamplesPerChannel:
			data = loadChunk(f, nChannels, channels, nSamples, sampleSize)
		else:
			# Determine chunk duration and number of chunks
			nSamplesPerChunk = math.floor(maxSamplesPerChunk/nChannels)*nChannels
			nChunks = math.floor(nSamples/nSamplesPerChunk)

			if verbose:
				print("nSamplesPerChannel ", nSamplesPerChannel)
				print("nSamplesPerChunk ", nSamplesPerChunk)

			# Preallocate memory
			data = np.zeros((nSamplesPerChannel,len(channels)), dtype=np.int16)

			if verbose:
				print("size data ", np.size(data, 0))

			# Read all chuncks
			i = 0
			for j in range(nChunks):
				d = loadChunk(f, nChannels, channels, nSamplesPerChunk/nChannels, sampleSize)
				m = np.size(d, 0)

				if m == 0:
					break

				data[i:i+m, :] = d
				i = i+m

			# If the data size is not a multiple of the chunk size, read the remainder
			remainder = nSamples - nChunks*nSamplesPerChunk
			if remainder != 0:
				d = loadChunk(f, nChannels, channels, remainder/nChannels, sampleSize)
				m = np.size(d, 0)

				if m != 0:
					data[i:i+m, :] = d

	return data



