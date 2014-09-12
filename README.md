pyhmm
=====

HMM Implementation in Python

This is a simple implementation of Discrete Hidden Markov Model developed as a teaching illustration for the NLP course. 

Usage:
=====

Please have a look at the file: test_hmm.py to get the sample code.

Example:

# instantiate the HMM by passing the file name of the model, model is in JSON format
# Sample model files are in the models subdirectory
model_file_name = r"./models/coins1.json"
hmm = MyHmm(model_file_name)

# get the probability of a sequence of observations P(O|model) using forward algorithm
observations = ("Heads", "Tails", "Heads", "Heads", "Heads", "Tails")
prob_1 = hmm.forward(observations) 

# get the probability of a sequence of observations P(O|model) using backward algorithm
prob_2 = hmm.backward(observations) 

# get the hidden states using Viterbi algorithm
(prob, states) = hmm.viterbi(observations) 

# For unsupervised learning, compute model parameters using forward-backward Baum Welch algorithm
hmm.forward_backward(observations) # hmm.A will contain transition probability, hmm.B will have the emission probability and hmm.pi will have the starting distribution

