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

def make_trans_matrix():
	afile = ['annotation1.fa', 'annotation2.fa', 
					 'annotation3.fa', 'annotation4.fa', 'annotation5.fa']
	gfile = ['genome1.fa', 'genome2.fa', 
					 'genome3.fa', 'genome4.fa', 'genome5.fa']

	trans_probs = np.zeros(43*43).reshape(43,43)
	
	trans_dict = combined_trans_dict(afile, gfile)

	# Række = FRA
	# Kolonne = TIL

	trans_probs[0][0] = trans_dict['N_N']
	trans_probs[0][1] = trans_dict['N_ATG']
	trans_probs[0][4] = trans_dict['N_GTG']
	trans_probs[0][7] = trans_dict['N_TTG']
	
	trans_probs[1][2] = 1
	trans_probs[2][3] = 1
	trans_probs[4][5] = 1
	trans_probs[5][6] = 1
	trans_probs[7][8] = 1
	trans_probs[8][9] = 1
	trans_probs[3][10] = 1
	trans_probs[6][10] = 1
	trans_probs[9][10] = 1
	trans_probs[10][11] = 1
	trans_probs[11][12] = 1

	trans_probs[12][10] = trans_dict['C_C']
	trans_probs[12][13] = trans_dict['TAG_N']
	trans_probs[12][16] = trans_dict['TGA_N']
	trans_probs[12][19] = trans_dict['TAA_N']

	trans_probs[13][14] = 1
	trans_probs[14][15] = 1
	trans_probs[15][0] = 1
	trans_probs[16][17] = 1
	trans_probs[17][18] = 1
	trans_probs[18][0] = 1
	trans_probs[19][20] = 1
	trans_probs[20][21] = 1
	trans_probs[21][0] = 1

	trans_probs[0][22] = trans_dict['N_TTA']
	trans_probs[0][25] = trans_dict['N_CTA']
	trans_probs[0][28] = trans_dict['N_TCA']

	trans_probs[22][23] = 1
	trans_probs[23][24] = 1
	trans_probs[24][31] = 1
	trans_probs[25][26] = 1
	trans_probs[26][27] = 1
	trans_probs[27][31] = 1
	trans_probs[28][29] = 1
	trans_probs[29][30] = 1
	trans_probs[30][31] = 1
	trans_probs[31][32] = 1
	trans_probs[32][33] = 1

	trans_probs[33][31] = trans_dict['R_R']
	trans_probs[33][34] = trans_dict['CAT_N']
	trans_probs[33][37] = trans_dict['CAC_N']
	trans_probs[33][40] = trans_dict['CAA_N']

	trans_probs[34][35] = 1
	trans_probs[35][36] = 1
	trans_probs[36][0] = 1
	trans_probs[37][38] = 1
	trans_probs[38][39] = 1
	trans_probs[39][0] = 1
	trans_probs[40][41] = 1
	trans_probs[41][42] = 1
	trans_probs[42][0] = 1


	# print("Transitions not zero or one")
	# it = np.nditer(trans_probs, flags=['multi_index'])
	# while not it.finished:
	# 	if (it[0] > 0 and it[0] < 1):
	# 		print("<%s> %f" % (it.multi_index, it[0]*100), "%")
	# 	it.iternext()

	# print()
	# print("All one transitions")
	# it = np.nditer(trans_probs, flags=['multi_index'])
	# while not it.finished:
	# 	if (it[0] == 1):
	# 		print("<%s> %f" % (it.multi_index, it[0]))
	# 	it.iternext()


	emit_probs = np.zeros(43 * 4).reshape(43, 4)
	coding_emissions, rev_coding_emissions, N_results = count_emissions(afile, gfile)
	
	emit_probs[0] = N_results
	emit_probs[1] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[2] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[3] = [0.0, 0.0, 1.0, 0.0] # G
	emit_probs[4] = [0.0, 0.0, 1.0, 0.0] # G
	emit_probs[5] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[6] = [0.0, 0.0, 1.0, 0.0] # G
	emit_probs[7] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[8] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[9] = [0.0, 0.0, 1.0, 0.0] # G

	emit_probs[10] = coding_emissions[0]
	emit_probs[11] = coding_emissions[1]
	emit_probs[12] = coding_emissions[2]

	emit_probs[13] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[14] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[15] = [0.0, 0.0, 1.0, 0.0] # G
	emit_probs[16] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[17] = [0.0, 0.0, 1.0, 0.0] # G
	emit_probs[18] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[19] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[20] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[21] = [1.0, 0.0, 0.0, 0.0] # A

	emit_probs[22] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[23] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[24] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[25] = [0.0, 1.0, 0.0, 0.0] # C
	emit_probs[26] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[27] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[28] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[29] = [0.0, 1.0, 0.0, 0.0] # C
	emit_probs[30] = [1.0, 0.0, 0.0, 0.0] # A

	emit_probs[31] = rev_coding_emissions[0]
	emit_probs[32] = rev_coding_emissions[1]
	emit_probs[33] = rev_coding_emissions[2]

	emit_probs[34] = [0.0, 1.0, 0.0, 0.0] # C
	emit_probs[35] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[36] = [0.0, 0.0, 0.0, 1.0] # T
	emit_probs[37] = [0.0, 1.0, 0.0, 0.0] # C
	emit_probs[38] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[39] = [0.0, 1.0, 0.0, 0.0] # C
	emit_probs[40] = [0.0, 1.0, 0.0, 0.0] # C
	emit_probs[41] = [1.0, 0.0, 0.0, 0.0] # A
	emit_probs[42] = [1.0, 0.0, 0.0, 0.0] # A

	init_probs = np.zeros(43).astype('float')
	init_probs[0] = 1.0


	return trans_probs, emit_probs, init_probs


	
def main():
	# count_genome(["lol.txt"])
	annotations = ['annotation1.fa', 'annotation2.fa', 
								 'annotation3.fa', 'annotation4.fa', 'annotation5.fa']
	genomes = ['genome1.fa', 'genome2.fa', 
						 'genome3.fa', 'genome4.fa', 'genome5.fa']					 			 
	# count_genome(annotations)


	# count_emissions(annotations, genomes)
	make_trans_matrix()


if __name__ == "__main__":
	main()
