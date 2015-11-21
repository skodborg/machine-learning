import numpy as np
import time
import count_genome as count
import itertools
import compare_anns as comp

np.seterr(divide='ignore')


def viterbi_logspace_backtrack(trained_probs, input_genome, outputfilename):
	observables = {'A':0, 'C':1, 'G':2, 'T':3}

	states = {'1':0, '2': 1, '3':2, '4':3, '5':4, '6':5, '7':6,
						'8':7, '9': 8, '10':9, '11':10, '12':11, '13':12,
						'14':13,'15':14, '16': 15, '17': 16, '18': 17, '19': 18,
						'20': 19, '21': 20, '22': 21, '23': 22, '24': 23, '25': 24,
						'26': 25, '27': 26, '28': 27, '29': 28, '30': 29, '31': 30,
						'32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36,
						'38': 37, '39': 38, '40': 39, '41': 40, '42': 41, '43': 42}

	print("Loading trained matrices")

	trans_probs, emit_probs, init_probs = trained_probs

	print("Loading input genome: " + str(input_genome))
	g = open(input_genome, 'r')
	g.readline()
	genome = g.read().replace('\n', '')

	test_observations = genome

	print("Running viterbi_logspace_backtrack...")

	# BASIS
	idx_first_obs = observables.get(test_observations[0])
	omega = np.empty([len(test_observations),trans_probs.shape[0]])
	omega[0] = np.log([init_probs]) + np.log(emit_probs[:,idx_first_obs])
	max_vector = np.zeros(len(states))

	# RECURSIVE
	for obs in range(1, len(test_observations)):
	
		# iterating through all states to generate next col in omega
		for i, _ in enumerate(states):

			# find transition probabilities from every state to this current state
			trans_to_state_i = trans_probs[:,i]
			
			# fetch previous col in omega
			prev_omega_col = omega[obs-1]

			# find the max probability that this state will follow from the prev col
			state_i_max_prob = np.max(prev_omega_col + np.log(trans_to_state_i))

			# save for multiplying with emission probabilities to determine omega col
			max_vector[i] = state_i_max_prob
			
		# get idx of current observation to use with defined matrix data structures
		idx_curr_obs = observables.get(test_observations[obs])

		# get emission probabilities of current observation for all states
		emit_probs_curr_obs = emit_probs[:,idx_curr_obs]

		# create and add the new col to the omega table
		new_omega_col = np.log(emit_probs_curr_obs) + max_vector

		omega[obs] = new_omega_col

		if (obs % 100000 is 0):
			print(str(obs) + ": max in last col of omega: " + str(np.max(omega[obs])))
		
	# natural log to the most likely probability when all the input is processeds  
	print("max in last col of omega: %s" % np.max(omega[-1]))


	# BACKTRACKING
	print("Running backtracking...")

	N = len(test_observations)-1  # off-by-one correction for indexing into lists
	K = len(states)
	z = np.zeros(len(test_observations))
	z[N] = np.argmax(omega[len(omega)-1], axis=0)
	max_vector = np.zeros(K)
	
	# n descending from N-1 to 0 inclusive
	for n in range(N-1, -1, -1):
		if (n % 100000 is 0):
			print(n)
	
		for k in range(0, K):
			# only for matching pseudocode easily
			x = test_observations

			# matrix data structure index of observation
			idx_obs = observables.get(x[n+1])

			# probability of observing x[n+1] in state z[n+1]
			p_xn1_zn1 = emit_probs[z[n+1]][idx_obs]

			# our omega table indexing is flipped compared to the pseudocode alg.
			omega_kn = omega[n][k]
 
			# get transitions from state k to state z[n+1]
			p_zn1_k = trans_probs[k,z[n+1]]

			# add product to max_vector
			max_vector[k] = np.log(p_xn1_zn1) + omega_kn + np.log(p_zn1_k)
			
		# set z[n] to arg max of max_vector
		z[n] = np.argmax(max_vector)

	# add one to correspond to actual states rather than indexes into 'states'
	z = z + 1
	# print(z)

	print("Converting output from states to annotations...")
	result = []
	for i in range(0, len(z)):
		if z[i] == 1:
			result.append("N")
		elif z[i] > 22:
			result.append("R")
		else:
			result.append("C")

	f = open(outputfilename, "w")
	for s in result:
		f.write(s)
	f.close()

def cross_validate_viterbi():
	annotationfiles = ['annotation1.fa', 'annotation2.fa', 
										 'annotation3.fa', 'annotation4.fa', 'annotation5.fa']
	genomefiles = ['genome1.fa', 'genome2.fa', 
								 'genome3.fa', 'genome4.fa', 'genome5.fa']

	# zipping genomes and annotations to list of pairwise tuples
	genomes_annots_zipped = list(zip(genomefiles, annotationfiles))

	# iterating through all combinations of 4 tuples of the list
	for i, x in enumerate(itertools.combinations(genomes_annots_zipped, 4)):
		# unzipping the training sets
		training_genomes, training_annots = list(zip(*x))
		
		# extracting the remaining validation set
		valdation_genome, validation_annot = genomes_annots_zipped[4-i]
		
		# training
		outputfilename = "result_annot_"+str(5-i)+".fa"
		trained_probs = count.get_trained_matrices(training_annots, training_genomes)

		# create annotations for manual validation
		viterbi_logspace_backtrack(trained_probs, valdation_genome, outputfilename)
			
def annotate_data():
	annotationfiles = ['annotation1.fa', 'annotation2.fa', 
										 'annotation3.fa', 'annotation4.fa', 'annotation5.fa']
	genomefiles = ['genome1.fa', 'genome2.fa', 
								 'genome3.fa', 'genome4.fa', 'genome5.fa']

	unknown_genomes_files = ['genome6.fa', 'genome7.fa', 
								 					 'genome8.fa', 'genome9.fa', 'genome10.fa']

	trained_probs = count.get_trained_matrices(annotationfiles, genomefiles)
	for i, genome in enumerate(unknown_genomes_files):
		nr_genome = 10 - i
		print("Predicting " + genome)
		outputfilename = "result_annot_"+genome
		viterbi_logspace_backtrack(trained_probs, genome, outputfilename)

def main():
	# viterbi_logspace_backtrack()
	
	# cross_validate_viterbi()

	annotate_data()



if __name__ == "__main__":
	main()

