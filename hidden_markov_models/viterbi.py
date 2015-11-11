import numpy as np

test_observations = "GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA"

observables = {'A':0, 'C':1, 'G':2, 'T':3}
states = {'1':0, '2': 1, '3':2, '4':3, '5':4, '6':5, '7':6}

init_probs = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]

trans_probs = np.array([[0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00], 
                        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], 
                        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00], 
                        [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00], 
                        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00], 
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00], 
                        [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00]])

emit_probs = np.array([[0.30, 0.25, 0.25, 0.20], 
                       [0.20, 0.35, 0.15, 0.30], 
                       [0.40, 0.15, 0.20, 0.25], 
                       [0.25, 0.25, 0.25, 0.25], 
                       [0.20, 0.40, 0.30, 0.10], 
                       [0.30, 0.20, 0.30, 0.20], 
                       [0.15, 0.30, 0.20, 0.35]])

def viterbi_backtrack():  
  # BASIS
  idx_first_obs = observables.get(test_observations[0])
  omega = [init_probs] * emit_probs[:,idx_first_obs]


  # RECURSIVE
  for obs in range(1, len(test_observations)):

    max_vector = []
    # iterating through all states to generate next col in omega
    for i, _ in enumerate(states):

      # find transition probabilities from every state to this current state
      trans_to_state_i = trans_probs[:,i]
      
      # fetch previous col in omega
      prev_omega_col = omega[-1]

      # find the max probability that this state will follow from the prev col
      state_i_max_prob = np.max(prev_omega_col * trans_to_state_i)

      # save for multiplying with emission probabilities to determine omega col
      max_vector.append(state_i_max_prob)

    # get idx of current observation to use with defined matrix data structures
    idx_curr_obs = observables.get(test_observations[obs])
    
    # get emission probabilities of current observation for all states
    emit_probs_curr_obs = emit_probs[:,idx_curr_obs]
    
    # create and add the new col to the omega table
    new_omega_col = emit_probs_curr_obs * max_vector
    omega = np.append(omega, [new_omega_col], axis=0)

  # natural log to the most likely probability when all the input is processeds
  print(np.log(np.max(omega[-1])))


  # BACKTRACKING

  N = len(test_observations)-1  # off-by-one correction for indexing into lists
  K = len(states)
  z = np.zeros(len(test_observations))
  z[N] = np.argmax(omega[len(omega)-1], axis=0)
  
  # n descending from N-1 to 0 inclusive
  for n in range(N-1, -1, -1):
  
    max_vector = []
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
      max_vector.append(p_xn1_zn1 * omega_kn * p_zn1_k)
    
    # set z[n] to arg max of max_vector
    z[n] = np.argmax(max_vector)

  # add one to correspond to actual states rather than indexes into 'states'
  z = z + 1
  print(z)


def main():
  viterbi_backtrack()

if __name__ == "__main__":
  main()

