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
	coding_emissions = [[0.0, 0.0, 0.0, 0.0],
											[0.0, 0.0, 0.0, 0.0],
											[0.0, 0.0, 0.0, 0.0]]
	for i in range(0,5):
		with open(annotations[i], 'r') as a, open(genomes[i], 'r') as g:
			# remove first line with description in every file
			a.readline()
			g.readline()
			annot = a.read().replace('\n', '')
			genome = g.read().replace('\n', '')

			idx_start_codon = [m.start() for m in re.finditer('(N|R)CCC', annot)]
			idx_stop_codon = [m.start() for m in re.finditer('CCC(N|R)', annot)]

			
			for i in range(0,len(idx_start_codon)):
				start_idx = idx_start_codon[i] + 4
				stop_idx = idx_stop_codon[i]
				
				coding_substring = genome[start_idx:stop_idx]

				for i, c in enumerate(coding_substring):
					# coding_substring is always a multiple of 3 long
					coding_emissions[i%3][observables.get(c)] += 1

	for i in range(0,len(coding_emissions)):
		curr_state_sum = sum(coding_emissions[i])
		for j in range(0, len(coding_emissions[i])):
			coding_emissions[i][j] = coding_emissions[i][j] / curr_state_sum

	# print("C emissions, state 1-2-3, emissions A-C-G-T")
	# print(coding_emissions)
	# print()


	rev_coding_emissions = [[0.0, 0.0, 0.0, 0.0],
													[0.0, 0.0, 0.0, 0.0],
													[0.0, 0.0, 0.0, 0.0]]
	for i in range(0,5):
		with open(annotations[i], 'r') as a, open(genomes[i], 'r') as g:
			# remove first line with description in every file
			a.readline()
			g.readline()
			annot = a.read().replace('\n', '')
			genome = g.read().replace('\n', '')

			idx_start_codon = [m.start() for m in re.finditer('(N|C)RRR', annot)]
			idx_stop_codon = [m.start() for m in re.finditer('RRR(N|C)', annot)]

			
			for i in range(0,len(idx_start_codon)):
				start_idx = idx_start_codon[i] + 4
				stop_idx = idx_stop_codon[i]
				
				rev_coding_substring = genome[start_idx:stop_idx]

				for i, c in enumerate(rev_coding_substring):
					# rev_coding_substring is always a multiple of 3 long
					rev_coding_emissions[i%3][observables.get(c)] += 1

	for i in range(0,len(rev_coding_emissions)):
		curr_state_sum = sum(rev_coding_emissions[i])
		for j in range(0, len(rev_coding_emissions[i])):
			rev_coding_emissions[i][j] = rev_coding_emissions[i][j] / curr_state_sum

	# print("R emissions, state 1-2-3, emissions A-C-G-T")
	# print(rev_coding_emissions)
	# print()

			
	N_results = [0,0,0,0]
	for j, c in enumerate(annot):
		a_char = annot[j]
		g_char = genome[j]
		idx = observables.get(g_char)
		if (a_char is 'N'):
			N_results[idx] = N_results[idx] + 1

	N_sum = np.sum(N_results).astype(float)
	N_results = N_results / N_sum
	# print("N emissions, A-C-G-T")
	# print(N_results)

	return coding_emissions, rev_coding_emissions, N_results


def count_stop_codons(annotationfiles, genomefiles):
	all_stop_codons_dict = {}
	CCC_sum = 0.0

	for i in range(0,5):
		with open(annotationfiles[i], "r") as annotationfile, \
				 open(genomefiles[i], "r") as genomefile:

			annotationfile.readline()
			genomefile.readline()
			annot = annotationfile.read().replace('\n', '')
			genome = genomefile.read().replace('\n', '')

			# find all consecutive C triplets
			CCC = len([m.start() for m in re.finditer('CCC', annot)])
			
			# find number of genes by counting start-codons
			NCCC = len([m.start() for m in re.finditer('(N|R)CCC', annot)])

			# subtract start- and stop-codons from total number of CCC triplets
			CCC_sum += CCC - (2 * NCCC)

			# currfile_start_codons = []
			idx_stop_codon = [m.start() for m in re.finditer('CCC(N|R)', annot)]
			for i in idx_stop_codon:
				curr_codon = ""
				for c in range(0,3):
					curr_codon += genome[i+c]
				if curr_codon in all_stop_codons_dict:
					all_stop_codons_dict[curr_codon] += 1
				else:
					all_stop_codons_dict[curr_codon] = 1
		
	# print(all_stop_codons_dict)

	# for key, value in all_stop_codons_dict.items():
	# 	print("N to "+key+": %s" % (value / NX_sum))
	# print("N back to N: %s" % (NN_sum / NX_sum))

	TAG_N = all_stop_codons_dict['TAG'] / CCC_sum
	TGA_N = all_stop_codons_dict['TGA'] / CCC_sum
	TAA_N = all_stop_codons_dict['TAA'] / CCC_sum
	C_C = 1 - (TAG_N + TGA_N + TAA_N)

	result = {'TAG_N':TAG_N, 'TGA_N':TGA_N, 'TAA_N':TAA_N, 'C_C':C_C}
	return result



def count_start_codons(annotationfiles, genomefiles):
	all_start_codons_dict = {}
	NX_sum = 0.0

	for i in range(0,5):
		with open(annotationfiles[i], "r") as annotationfile, \
				 open(genomefiles[i], "r") as genomefile:

			annotationfile.readline()
			genomefile.readline()
			annot = annotationfile.read().replace('\n', '')
			genome = genomefile.read().replace('\n', '')

			# transitions N to anything (total transitions out of N)
			NX = len([m.start() for m in re.finditer('(?=N.)', annot)])
			NX_sum += NX

			# currfile_start_codons = []
			idx_start_codon = [m.start() for m in re.finditer('(N|R)CCC', annot)]
			for i in idx_start_codon:
				curr_codon = ""
				for c in range(0,3):
					curr_codon += genome[(i+1)+c]
				if curr_codon in all_start_codons_dict:
					all_start_codons_dict[curr_codon] += 1
				else:
					all_start_codons_dict[curr_codon] = 1
		
	# print(all_start_codons_dict)

	# for key, value in all_start_codons_dict.items():
	# 	print("N to "+key+": %s" % (value / NX_sum))
	# print("N back to N: %s" % (NN_sum / NX_sum))

	N_ATG = all_start_codons_dict['ATG'] / NX_sum
	N_GTG = all_start_codons_dict['GTG'] / NX_sum
	N_TTG = all_start_codons_dict['TTG'] / NX_sum

	result = {'N_ATG':N_ATG, 'N_GTG':N_GTG, 'N_TTG':N_TTG}
	return result

def count_reverse_stop_codons(annotationfiles, genomefiles):
	all_rev_stop_codons_dict = {}
	NX_sum = 0.0

	for i in range(0,5):
		with open(annotationfiles[i], "r") as annotationfile, \
				 open(genomefiles[i], "r") as genomefile:

			annotationfile.readline()
			genomefile.readline()
			annot = annotationfile.read().replace('\n', '')
			genome = genomefile.read().replace('\n', '')

			# transitions N to anything (total transitions out of N)
			NX = len([m.start() for m in re.finditer('(?=N.)', annot)])
			NX_sum += NX

			
			idx_stop_codon = [m.start() for m in re.finditer('(N|C)RRR', annot)]
			for i in idx_stop_codon:
				curr_codon = ""
				for c in range(0,3):
					curr_codon += genome[(i+1)+c]
				if curr_codon in all_rev_stop_codons_dict:
					all_rev_stop_codons_dict[curr_codon] += 1
				else:
					all_rev_stop_codons_dict[curr_codon] = 1
		
	# print(all_rev_stop_codons_dict)

	# for key, value in all_rev_stop_codons_dict.items():
	# 	print("N to "+key+": %s" % (value / NX_sum))
	# print("N back to N: %s" % (NN_sum / NX_sum))

	N_TTA = all_rev_stop_codons_dict['TTA'] / NX_sum
	N_CTA = all_rev_stop_codons_dict['CTA'] / NX_sum
	N_TCA = all_rev_stop_codons_dict['TCA'] / NX_sum

	result = {'N_TTA':N_TTA, 'N_CTA':N_CTA, 'N_TCA':N_TCA}
	return result

def count_reverse_start_codons(annotationfiles, genomefiles):
	all_rev_start_codons_dict = {}
	RRR_sum = 0.0

	for i in range(0,5):
		with open(annotationfiles[i], "r") as annotationfile, \
				 open(genomefiles[i], "r") as genomefile:

			annotationfile.readline()
			genomefile.readline()
			annot = annotationfile.read().replace('\n', '')
			genome = genomefile.read().replace('\n', '')

			# find all consecutive C triplets
			RRR = len([m.start() for m in re.finditer('RRR', annot)])
			
			# find number of genes by counting start-codons
			NRRR = len([m.start() for m in re.finditer('(N|C)RRR', annot)])

			# subtract start- and stop-codons from total number of RRR triplets
			RRR_sum += RRR - (2 * NRRR)

			# currfile_start_codons = []
			idx_start_codon = [m.start() for m in re.finditer('RRR(N|C)', annot)]
			for i in idx_start_codon:
				curr_codon = ""
				for c in range(0,3):
					curr_codon += genome[i+c]
				if curr_codon in all_rev_start_codons_dict:
					all_rev_start_codons_dict[curr_codon] += 1
				else:
					all_rev_start_codons_dict[curr_codon] = 1
		
	# print(all_rev_start_codons_dict)

	# for key, value in all_rev_start_codons_dict.items():
	# 	print("N to "+key+": %s" % (value / NX_sum))
	# print("N back to N: %s" % (NN_sum / NX_sum))

	CAT_N = all_rev_start_codons_dict['CAT'] / RRR_sum
	CAC_N = all_rev_start_codons_dict['CAC'] / RRR_sum
	CAA_N = all_rev_start_codons_dict['CAA'] / RRR_sum
	R_R = 1 - (CAT_N + CAC_N + CAA_N)

	result = {'CAT_N':CAT_N, 'CAC_N':CAC_N, 'CAA_N':CAA_N, 'R_R':R_R}
	return result

def combined_trans_dict(annotations, genomes):
	start_codons = count_start_codons(annotations, genomes)
	stop_codons = count_stop_codons(annotations, genomes)
	rev_stop_codons = count_reverse_stop_codons(annotations, genomes)
	rev_start_codons = count_reverse_start_codons(annotations, genomes)

	temp = {**start_codons, **rev_stop_codons}
	sum_of_trans_out_of_N = 0.0
	for k, v in temp.items():
		sum_of_trans_out_of_N += v

	N_N = 1 - sum_of_trans_out_of_N
	
	return {**start_codons, **stop_codons, **rev_start_codons, **rev_stop_codons, 'N_N':N_N}

	
def main():
	# count_genome(["lol.txt"])
	annotations = ['annotation1.fa', 'annotation2.fa', 
								 'annotation3.fa', 'annotation4.fa', 'annotation5.fa']
	genomes = ['genome1.fa', 'genome2.fa', 
						 'genome3.fa', 'genome4.fa', 'genome5.fa']					 			 
	# count_genome(annotations)


	count_emissions(annotations, genomes)


	
	# count_start_codons(annotations, genomes)
	# count_stop_codons(annotations, genomes)
	# count_reverse_stop_codons(annotations, genomes)
	# count_reverse_start_codons(annotations, genomes)

	# print(combined_trans_dict(annotations, genomes))


if __name__ == "__main__":
	main()
