MyHmm
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

Note
=====
For long sequences of observations the HMM computations may result in underflow. In particular when training the HMM with multiple input sequences (for example during speech recognition tasks) often results in underflows. The module myhmm_scaled can be used instead of myhmm to train the HMM for long sequences. It is important to note the following. <br/>
1. The implementation is as per Rabiner's paper with the errata addressed. See: http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html <br>
2. forward_scaled implements the scaled forward algorithm and returns log(P(Observations)) instead of P(Observations). If P(O) >= minimum floating point number that can be represented, then we can get back P(O) by math.exp(log_p). But if P(O) is smaller it will cause underflows. <br/>
3. backward_scaled implements the scaled version of backward algorithm. This is implemented for the sole purpose of computing zi, gamma for training using Baum Welch algorithm. It should always be called after executing the forward procedure (as the clist needs to be set up) <br/>
4. forward_backward_multi_scaled implements the scaled training procedure that supports multiple observation sequences. <br/>

Usage:
======
from myhmm_scaled import MyHmmScaled

model_file_name = r"./models/coins1.json"
hmm = MyHmmScaled(model_file_name)

\# compute model parameters using forward-backward Baum Welch algorithm with scaling (Refer Rabiner) <br />  
hmm.forward_backward_multi_scaled(observations) 
\# hmm.A will contain transition probability, hmm.B will have the emission probability and hmm.pi will have the starting distribution

\# get the log probability of a sequence of observations log P(O|model) using forward_scaled algorithm <br />    
observations = [("Heads", "Tails", "Heads", "Heads", "Heads", "Tails"), ("Tails", "Tails", "Tails", "Heads", "Heads", "Tails")] <br />  
log_prob_1 = hmm.forward_scaled(observations[0])  <br />  
log_prob_2 = hmm.forward_scaled(observations[1])  <br />  
