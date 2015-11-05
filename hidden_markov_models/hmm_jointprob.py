#
# hmm_jointprob.py
#
# Christian Storm Pedersen, 10-feb-2014

import sys
import math

# Model parameters from ML exercise HMM, Fall 2014

observables = {'A':0, 'C':1, 'G':2, 'T':3}
states = {'1':0, '2': 1, '3':2, '4':3, '5':4, '6':5, '7':6}

init_probs = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]

trans_probs = [[0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00], \
               [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], \
               [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00], \
               [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00], \
               [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00], \
               [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00], \
               [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00]]

emit_probs = [[0.30, 0.25, 0.25, 0.20], \
              [0.20, 0.35, 0.15, 0.30], \
              [0.40, 0.15, 0.20, 0.25], \
              [0.25, 0.25, 0.25, 0.25], \
              [0.20, 0.40, 0.30, 0.10], \
              [0.30, 0.20, 0.30, 0.20], \
              [0.15, 0.30, 0.20, 0.35]]

# Function for computing the joint probability 
def compute_joint_prob(x, z):
    p = init_probs[z[0]] * emit_probs[z[0]][x[0]]
    for i in range(1, len(x)):
        p = p * trans_probs[z[i-1]][z[i]] * emit_probs[z[i]][x[i]]
    return p

# Function for computing the joint probability in log space
def compute_joint_log_prob(x, z):
    logp = math.log(init_probs[z[0]]) + math.log(emit_probs[z[0]][x[0]])
    for i in range(1, len(x)):
        logp = logp + math.log(trans_probs[z[i-1]][z[i]]) + math.log(emit_probs[z[i]][x[i]])
    return logp

# Get sequence of observables and hidden states from the commandline
x = [observables[c] for c in list(sys.argv[1])]
z = [states[c] for c in list(sys.argv[2])]

if len(x) != len(z):
    print "The two sequences are of different length!"
    sys.exit(1)

print "    P(x,z) = ", compute_joint_prob(x, z)
print "log P(x,z) = ", compute_joint_log_prob(x, z)
