Training the DRL agent in general resembles the normal Q-learning approach but has some differences.


When agent is about to select an action in an e-greedy\cite{egreedy} manner, instead of calculating the current $Q(s,a)$ the estimation of the $\hat{Q}^*(s,a)$ is obtained from the MLP. In this moment, update of the values of the state-space can not be observed.
Update of the network happens after the decision is taken and when there is enough training data in memory.


Because of the fact that initial estimations of the $Q$ might be far from optimal due to random decisions of actions, it is beneficial for the system to forget some of the estimates, and forget them from the memory. That is why the training of the MLP begins, when the memory has enough samples and the training can be done in batches with fixed size of the stored experiences.


Moreover, after achieving big enough pool of experience, changing the weights of the network by training and constantly using the same network to estimate the $\hat{Q}^*(s,a)$ has high chances to cause instability in training \cite{dqn_two_networks}.


To solve this issue two MLP networks with the same architectures are used. One to constantly return the estimate which is used to select an action, and the other to constantly train using the gathered experience from the batches. The estimating network is updated to the state of the constantly trained network with certain frequency. 
