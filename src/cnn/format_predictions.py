import numpy as np 


def get_predictions_indexes(data, predictions, window_size=12.8, fs=1250, threshold=0.5):
	window_pts = window_size * fs
	pred_indexes = []

	for i_pred, pred in enumerate(predictions):
		if (pred >= threshold):
			ini = i_pred * window_pts
			end = ini + window_pts - 1

			pred_indexes.append(np.array([ini, end]))


	return pred_indexes


