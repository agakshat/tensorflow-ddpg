This is an implementation of the Deep Deterministic Policy Gradients algorithm in Tensorflow, using the Keras library as the frontend for ease of use. 

It has been tested with the MountainCarContinuous-v0 and the Pendulum-v0 environment from the OpenAI gym.

Requires tensorflow, keras, gym, numpy installed

The command to run is: `python DDPG.py`.

A lot of parameters like learning rate, batch size, environment etc. can be passed through the command line, have a look inside the DDPG.py folder to see the argument names.
