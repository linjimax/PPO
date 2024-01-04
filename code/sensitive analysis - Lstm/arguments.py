"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse

def get_args():
	"""
		Description:
		Parses arguments at command line.

		Parameters:
			None

		Return:
			args - the arguments parsed
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='retrain')              # can be 'train' or 'test' or 'retrain or 'manytest'
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')        # your acitic model filename

	args = parser.parse_args()

	return args