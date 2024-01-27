See debug.py to see example of how everything can be run. Summary of contents:

1. *environments* contains the Environments (currently only a single environment, with a simple one-dimensional state space and uniformly distributed transitions, and where one end of state space is always prefered for reward maximization, so adversarial transition function can be computed analytically).
2. *policies* contains different policies I am experimenting with
3. *models* contains nn.Module code for modelling the Q / beta / w functions (along with the critic functions for minimax methods)
4. *learners* contains the learning algorihtms (currently have implemented minimax algorithm for estimating Q/beta)
5. *utils* contains some useful generic utilities
