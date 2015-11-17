import numpy as np
import re

observables = {'A':0, 'C':1, 'G':2, 'T':3}

def find_regex_occ(regex, data, trailing=False):
	pass

def count_genome(argfiles):

	countings = []
	# files = ['annotation1.fa', 'annotation2.fa', 
	# 				 'annotation3.fa', 'annotation4.fa', 'annotation5.fa']

	overlapping_substring = 'NN'
	for f in argfiles:
		data = ''
		with open(f, "r") as genomefile:
			subresults = []
			data = genomefile.read().replace('\n', '')

			# transitions N to N
			NN = len([m.start() for m in re.finditer('(?=NN)', data)])
			subresults.append(NN)

			# transitions N to anything (total transitions out of N)
			NX = len([m.start() for m in re.finditer('(?=N.)', data)])
			subresults.append(NX)

			# four consecutive C's (corresponding to C's after initial three)
			# right most box on slide
			CCCC = len([m.start() for m in re.finditer('(?=CCCC)', data)])
			subresults.append(CCCC)

			# hacky, only works because there is no 'NCCCN'-occurences in our files
			CCCX = len([m.start() for m in re.finditer('(?=CCC.)', data)])
			subresults.append(CCCX)

			countings.append(subresults)

	countings = np.array(countings).astype(float)
	
	NN = countings[:,0]
	NX = countings[:,1]
	CCCC = countings[:,2]
	CCCX = countings[:,3]

	avg_NNs = np.average(NN / NX)
	print(avg_NNs)
	print(1.0 - avg_NNs)

	avg_CCCCs = np.average(CCCC / CCCX)
	print(avg_CCCCs)
	print(1.0 - avg_CCCCs)


def count_emissions(annotations, genomes):
	for i in range(0,1):
		with open(annotations[i], 'r') as a, open(genomes[i], 'r') as g:
			# remove first line with description in every file
			a.readline()
			g.readline()
			annot = a.read().replace('\n', '')
			genome = g.read().replace('\n', '')

			C_results = [0,0,0,0]
			N_results = [0,0,0,0]
			for j, c in enumerate(annot):
				a_char = annot[j]
				g_char = genome[j]
				idx = observables.get(g_char)
				if (a_char is 'C'):
					C_results[idx] = C_results[idx] + 1
				else:
					# assuming an R observation is an N for now
					N_results[idx] = N_results[idx] + 1
					
			print(C_results)
			print(N_results)

			C_sum = np.sum(C_results).astype(float)
			N_sum = np.sum(N_results).astype(float)

			C_results = C_results / C_sum
			N_results = N_results / N_sum

			print(C_results)
			print(N_results)



	
def main():
  # count_genome(["lol.txt"])
	annotations = ['annotation1.fa', 'annotation2.fa', 
								 'annotation3.fa', 'annotation4.fa', 'annotation5.fa']
	genomes = ['genome1.fa', 'genome2.fa', 
						 'genome3.fa', 'genome4.fa', 'genome5.fa']					 			 
  # count_genome(annotations)
	count_emissions(annotations, genomes)


if __name__ == "__main__":
  main()
