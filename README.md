pyhmm
=====

HMM Implementation in Python

This is a simple implementation of Discrete Hidden Markov Model developed as a teaching illustration for the NLP course. 

Usage:
=====

Please have a look at the file: test_hmm.py to get the sample code.

Example:

\# instantiate the HMM by passing the file name of the model, model is in JSON format <br />  

\# Sample model files are in the models subdirectory <br />  


model_file_name = r"./models/coins1.json"
hmm = MyHmm(model_file_name)

\# get the probability of a sequence of observations P(O|model) using forward algorithm <br />    

observations = ("Heads", "Tails", "Heads", "Heads", "Heads", "Tails") <br />  
prob_1 = hmm.forward(observations)  <br />  

\# get the probability of a sequence of observations P(O|model) using backward algorithm  <br />  
prob_2 = hmm.backward(observations)  <br />  

\# get the hidden states using Viterbi algorithm  <br />  
(prob, states) = hmm.viterbi(observations)  <br />  

\# For unsupervised learning, compute model parameters using forward-backward Baum Welch algorithm  <br />  
hmm.forward_backward(observations) # hmm.A will contain transition probability, hmm.B will have the emission probability and hmm.pi will have the starting distribution

