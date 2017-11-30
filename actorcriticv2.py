import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Add,Activation,Lambda
from keras.optimizers import Adam
import keras.backend as K

class ActorNetwork(object):
	"""
	Implements actor network
	"""
	def __init__(self,sess,state_dim,action_dim,lr,tau,action_bound):
		self.sess = sess
		K.set_session(sess)
		K.set_learning_phase(1)
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr =  lr
		self.tau = tau
		self.action_bound = action_bound
		self.mainModel,self.mainModel_weights,self.mainModel_state = self._build_model()
		self.targetModel,self.targetModel_weights,_ = self._build_model()
		self.action_gradient = tf.placeholder(tf.float32,[None,self.action_dim])
		self.params_grad = tf.gradients(self.mainModel.output,self.mainModel_weights,-self.action_gradient)
		grads = zip(self.params_grad,self.mainModel_weights)
		self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		input_obs = Input(shape=(self.state_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(64)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(self.action_dim)(h)
		h = Activation('tanh')(h)
		pred = Lambda(lambda h: h*self.action_bound)(h)
		#pred = tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=h).sample()
		model = Model(inputs=input_obs,outputs=pred)
		model.compile(optimizer='Adam',loss='categorical_crossentropy')
		return model,model.trainable_weights,input_obs

	def act(self,state,noise):
		act = self.mainModel.predict(state) + noise
		return act

	def predict_target(self,state):
		return self.targetModel.predict(state)

	def predict(self,state):
		return self.mainModel.predict(state)

	def update_target(self):
		wMain =  self.mainModel.get_weights()
		wTarget = self.targetModel.get_weights()
		for i in range(len(wMain)):
			wTarget[i] = self.tau*wMain[i] + (1-self.tau)*wTarget[i]
		self.targetModel.set_weights(wTarget)

	def train(self,state,action_grad):
		self.sess.run(self.optimize,feed_dict = {self.mainModel_state: state, self.action_gradient: action_grad})


class CriticNetwork(object):
	def __init__(self,sess,num_agents,state_dim,action_dim,lr,tau,gamma):
		self.sess = sess
		K.set_session(sess)
		K.set_learning_phase(1)
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr =  lr
		self.tau = tau
		self.num_agents = num_agents
		self.gamma  =  gamma
		self.mainModel,self.state,self.actions = self._build_model()
		self.targetModel,_,_ = self._build_model()
		self.action_grads  = tf.gradients(self.mainModel.output,self.actions)
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		temp1 = Dense(64)(h)
		action_abs = Dense(64)(input_actions)
		#action_abs = Activation('relu')(action_abs)
		#action_abs = BatchNormalization()(action_abs)
		h = Add()([temp1,action_abs])
		#h = Dense(64)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
		model.compile(optimizer='Adam',loss='mean_squared_error')
		return model,input_obs,input_actions

	def action_gradients(self,states,actions):
		return self.sess.run(self.action_grads,feed_dict={self.state: states, self.actions: actions})[0]

	def update_target(self):
		wMain =  self.mainModel.get_weights()
		wTarget = self.targetModel.get_weights()
		for i in range(len(wMain)):
			wTarget[i] = self.tau*wMain[i] + (1-self.tau)*wTarget[i]
		self.targetModel.set_weights(wTarget)

	def predict_target(self, state, actions):
		return self.targetModel.predict([state,actions])

	def predict(self, state, actions):
		x = np.ndarray((actions.shape[1],self.action_dim))
		for j in range(actions.shape[1]):
			x[j] = np.concatenate([y[j] for y in actions])
		return self.mainModel.predict([state,x])

	def train(self, state, actions, labels):
		self.mainModel.train_on_batch([state,actions],labels)
		#return self.predict(state,actions)