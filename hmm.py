import numpy as np
from matplotlib import pylab as plt
import scipy.stats

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import load_subjects as ls

# GENERATE TRAINING DATA
true_means = np.array([[40, 30], [15, 5], [25, 15], [15, 40]], dtype=np.float32)
true_durations = [10, 20, 10, 35]
true_Sigma = [[9.0, 0.0], [0.0, 1.0]]

observations = np.concatenate([
  scipy.stats.multivariate_normal(mean=mean, cov=true_Sigma).rvs(num_steps)
    for (mean, num_steps) in zip(true_means, true_durations)
]).astype(np.float32)

plt.plot(observations[:,0], observations[:,1], )

# LOAD DATA
# TODO(sss1): Restructure data into a tensor
# subjects = ls.load_all_data()

def sigmoid(x):
  """Computes the sigmoid function."""
  return tf.math.sigmoid(x)

def logit(x):
  """Computes the logit function, i.e. the logistic sigmoid inverse."""
  return - tf.math.log(1. / x - 1.)

# CONSTRUCT HMM WITH TRAINABLE EMISSION DISTRIBUTION
num_states = 4

# pi is distribution of the initial object of attention.
# We represent it in terms of logit-probabilities and then
# take sigmoids+normalize after optimization, as this is much simpler than
# constraining the probabilities themselves.
trainable_logit_pi_T = tf.Variable(logit(1/num_states), name='logit_pi_T', dtype='float32')

# Pi is the transition matrix of the object of attention.
# We represent it in terms of logits too.
switch_prob = 0.05
self_transition_prob = logit(1. - switch_prob)
switch_transition_prob = logit(switch_prob / (num_states-1))
trainable_logit_tau_T2 = tf.Variable(self_transition_prob, name='logit_tau_T2', dtype='float32')
trainable_logit_tau_DT = tf.Variable(switch_transition_prob, name='logit_tau_DT', dtype='float32')
trainable_logit_tau_D2 = tf.Variable(self_transition_prob, name='logit_tau_D2', dtype='float32')

# Sigma is the (diagonal) covariance matrix of gaze around the attended object.
trainable_Sigma = tf.Variable([2.0, 2.0], name='Sigma')

# We assume that the x- and y-components of the variance (around the attended
# object) are independent. In reality, this might not be the case, e.g., (if the
# object is moving diagonally), but we leave it to future work to improve this
# aspect of the model.
observation_distribution = tfd.MultivariateNormalDiag(loc=true_means,
                                                      scale_diag=trainable_Sigma
                                                     )

def construct_pi(logit_pi_T):
  """Constructs initial distribution from logit of pi_T."""
  pi_T = sigmoid(logit_pi_T)
  tau_D0 = (1 - pi_T) / (num_states-1)
  return [pi_T, tau_D0, tau_D0, tau_D0]

def construct_Pi(logit_tau_T2, logit_tau_DT, logit_tau_D2):
  """Constructs transition matrix from logits of tau_T2, tau_DT, and tau_D2."""
  tau_T2 = sigmoid(logit_tau_T2)
  tau_DT = sigmoid(logit_tau_DT)
  tau_D2 = sigmoid(logit_tau_D2)
  tau_TD = (1 - sigmoid(logit_tau_T2)) / (num_states-1)
  tau_DD = (1 - (sigmoid(logit_tau_DT) + sigmoid(logit_tau_D2))) / (num_states-2)
  return [[tau_T2, tau_TD, tau_TD, tau_TD],
          [tau_DT, tau_D2, tau_DD, tau_DD],
          [tau_DT, tau_DD, tau_D2, tau_DD],
          [tau_DT, tau_DD, tau_DD, tau_D2]]

def construct_hmm(logit_pi_T, logit_tau_T2, logit_tau_DT, logit_tau_D2):
  """Constructs the HMM model from its free parameters."""
  pi = construct_pi(logit_pi_T)
  Pi = construct_Pi(logit_tau_T2, logit_tau_DT, logit_tau_D2)
  return tfd.HiddenMarkovModel(
      initial_distribution=tfd.Categorical(probs=pi),
      transition_distribution=tfd.Categorical(probs=Pi),
      observation_distribution=observation_distribution,
      num_steps=len(observations))

# SPECIFY LOG-LIKELIHOOD TRAINING OBJECTIVE AND OPTIMIZER
def log_prob():
  hmm = construct_hmm(trainable_logit_pi_T,
                      trainable_logit_tau_T2,
                      trainable_logit_tau_DT,
                      trainable_logit_tau_D2)
  return hmm.log_prob(observations)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function(autograph=True)
def train_op():
  # Apply a gradient update.
  with tf.GradientTape() as tape:
    neg_log_prob = -log_prob()
  grads = tape.gradient(neg_log_prob, [trainable_logit_pi_T,
                                       trainable_logit_tau_T2,
                                       trainable_logit_tau_DT,
                                       trainable_logit_tau_D2,
                                       trainable_Sigma])
  optimizer.apply_gradients([(grads[0], trainable_logit_pi_T),
                             (grads[1], trainable_logit_tau_T2),
                             (grads[2], trainable_logit_tau_DT),
                             (grads[3], trainable_logit_tau_D2),
                             (grads[4], trainable_Sigma)])

  return neg_log_prob, trainable_logit_pi_T, trainable_logit_tau_T2, \
      trainable_logit_tau_DT, trainable_logit_tau_D2, trainable_Sigma

# FIT MODEL
loss_history = []
for step in range(2001):
  loss, logit_pi_T, logit_tau_T2, logit_tau_DT, logit_tau_D2, Sigma = [t.numpy() for t in train_op()]

  # Format probabilities nicely for printing.
  pi = [tau.numpy() for tau in construct_pi(logit_pi_T)]
  Pi = np.array([[tau.numpy() for tau in row] for row in construct_Pi(logit_tau_T2, logit_tau_DT, logit_tau_D2)])

  loss_history.append(loss)
  if step % 100 == 0:
    print("step {}: log prob {} Sigma {}\npi\n{}\nPi\n{}".format(step, -loss, Sigma, pi, Pi))
print("Inferred pi:\n{}".format(pi))
print("Inferred Pi:\n{}".format(Pi))
print("Inferred Sigma:\n{}".format(tf.linalg.diag(Sigma)))
print("True Sigma:\n{}".format(tf.math.sqrt(true_Sigma)))

# RUN FORWARD-BACKWARD ALGORITHM TO COMPUTE MARGINAL POSTERIORS
hmm = construct_hmm(logit_pi_T,
                    logit_tau_T2,
                    logit_tau_DT,
                    logit_tau_D2)
posterior_dists = hmm.posterior_marginals(observations)
posterior_probs = posterior_dists.probs_parameter().numpy()

def plot_state_posterior(ax, state_posterior_probs, title):
  ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | observations)')
  ax.set_ylim(0., 1.1)
  ax.set_ylabel('posterior probability')
  ax2 = ax.twinx()
  ln2 = ax2.plot(observations, c='black', alpha=0.3, label='observed observations')
  ax2.set_title(title)
  ax2.set_xlabel("time")
  lns = ln1+ln2
  labs = [l.get_label() for l in lns]
  ax.legend(lns, labs, loc=4)
  ax.grid(True, color='white')
  ax2.grid(False)

# PLOT MARGINAL POSTERIORS PROBABILITIES OF EACH STATE
fig = plt.figure(figsize=(10, 10))
plot_state_posterior(fig.add_subplot(2, 2, 1),
                     posterior_probs[:, 0],
                     title="state 0 (rate {:.2f})".format(true_means[0, 0]))
plot_state_posterior(fig.add_subplot(2, 2, 2),
                     posterior_probs[:, 1],
                     title="state 1 (rate {:.2f})".format(true_means[1, 0]))
plot_state_posterior(fig.add_subplot(2, 2, 3),
                     posterior_probs[:, 2],
                     title="state 2 (rate {:.2f})".format(true_means[2, 0]))
plot_state_posterior(fig.add_subplot(2, 2, 4),
                     posterior_probs[:, 3],
                     title="state 3 (rate {:.2f})".format(true_means[3, 0]))
plt.tight_layout()

plt.figure()
plt.plot(loss_history)

plt.show()
