import numpy as np
import gym
import tensorflow as tf
import random
from ReplayMemory import ReplayMemory

def build_summaries():
	episode_reward =   tf.Variable(0.)
	tf.summary.scalar("Reward",episode_reward)
	episode_ave_max_q = tf.Variable("episode_av_max_")
	tf.summary.scalar("QMaxValue",episode_ave_max_q)
	summary_vars = [episode_reward1,episode_ave_max_q]
	summary_vars = [episode_reward]
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars

def train(sess,env,args,actor,critic,actor_noise):
	
	summary_ops,summary_vars = build_summaries()
	init = tf.global_variables_initializer()
	sess.run(init)
	writer = tf.summary.FileWriter(args['summary_dir'],sess.graph)

	actor.update_target()
	critic.update_target()
	
	replayMemory = ReplayMemory(int(args['buffer_size']),int(args['random_seed']))

	for i in range(int(args['max_episodes'])):

		s = env.reset()
		episode_reward = 0
		episode_av_max_q = 0
		#if i%50==0:
			#actor.mainModel.save('results/mountainCar'+str(i)+'.h5')
			#print("Saving Model now")

		for j in range(int(args['max_episode_len'])):
			if args['render_env']:
				env.render()

			a = actor.act(np.reshape(s,(-1,actor.state_dim)),actor_noise())
			s2,r,done,_ = env.step(a[0])
			replayMemory.add(np.reshape(s,(actor.state_dim,)),np.reshape(a,(actor.action_dim,)),r,done,np.reshape(s2,(actor.state_dim,)))
			
			if replayMemory.size()>int(args['minibatch_size']):
				s_batch,a_batch,r_batch,d_batch,s2_batch = replayMemory.miniBatch(int(args['minibatch_size']))
				targetQ = critic.predict_target(s2_batch,actor.predict_target(s2_batch))
				yi = []
				for k in range(int(args['minibatch_size'])):
					if d_batch[k]:
						yi.append(r_batch[k])
					else:
						yi.append(r_batch[k] + critic.gamma*targetQ[k])
				critic.train(s_batch,a_batch,np.reshape(yi,(int(args['minibatch_size']),1)))

				actions_pred = actor.predict(s_batch)
				grads = critic.action_gradients(s_batch,actions_pred)
				actor.train(s_batch,grads)
				actor.update_target()
				critic.update_target()

			s = s2
			episode_reward += r
			if done:
				summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: episode_reward, summary_vars[1]: episode_av_max_q/float(j)})
				writer.add_summary(summary_str,i)
				writer.flush()
				print ('|Reward: {:d}| Episode: {:d}'.format(int(episode_reward),i))
				break

