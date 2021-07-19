import numpy as np
from matplotlib import pylab as plt
import scipy.stats

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import load_subjects as ls


trial_len = 600  # For now, we only consider the first 10s of each trial
trials_to_include = range(1, 11)  # Omit practice trials
num_trials = len(trials_to_include)
num_iters = 1001  # Number of training iterations
num_states = 7

# LOAD DATA
subjects = ls.load_all_data()
# observations = np.zeros((num_trials, trial_len, 2), dtype=np.float32)
# true_means = np.zeros((num_trials, trial_len, num_states, 2), dtype=np.float32)
# for trial_idx, trial in enumerate(trials_to_include):
#   observations[trial_idx, :, :] = subjects['0'].experiments['noshrinky'].datatypes['eyetrack'].trials[trial].data[:trial_len, 1:]
#   print(subjects['0'].experiments['noshrinky'].datatypes['eyetrack'].trials[trial].proportion_missing)
#   true_means[trial_idx, :, :, :] = subjects['0'].experiments['noshrinky'].datatypes['trackit'].trials[trial].object_positions[:, :trial_len, :].transpose(1, 0, 2)
trial = 1
observations = np.float32(subjects['0'].experiments['noshrinky'].datatypes['eyetrack'].trials[trial].data[:trial_len, 1:])
true_means = np.float32(subjects['0'].experiments['noshrinky'].datatypes['trackit'].trials[trial].object_positions[:, :trial_len, :].transpose(1, 0, 2))

def sigmoid(x):
  """Computes the sigmoid function."""
  return tf.math.sigmoid(x)

def logit(x):
  """Computes the logit function, i.e. the logistic sigmoid inverse."""
  return - tf.math.log(1. / x - 1.)

# CONSTRUCT HMM WITH TRAINABLE EMISSION DISTRIBUTION
# pi is distribution of the initial object of attention.
# We represent it in terms of logit-probabilities and then
# take sigmoids+normalize after optimization, as this is much simpler than
# constraining the probabilities themselves.
trainable_logit_pi_T = tf.Variable(logit(1/num_states), name='logit_pi_T', dtype='float32')

# Pi is the transition matrix of the object of attention.
# We represent it in terms of logits too.
switch_prob = 0.005
self_transition_prob = logit(1. - switch_prob)
switch_transition_prob = logit(switch_prob / (num_states-1))
trainable_logit_tau_T2 = tf.Variable(self_transition_prob, name='logit_tau_T2', dtype='float32')
trainable_logit_tau_DT = tf.Variable(switch_transition_prob, name='logit_tau_DT', dtype='float32')
trainable_logit_tau_D2 = tf.Variable(self_transition_prob, name='logit_tau_D2', dtype='float32')

# Sigma is the (diagonal) covariance matrix of gaze around the attended object.
Sigma_0 = 100.0
trainable_Sigma = tf.Variable([Sigma_0, Sigma_0], name='Sigma')

# We assume that the x- and y-components of the variance (around the attended
# object) are independent. In reality, this might not be the case, e.g., (if the
# object is moving diagonally), but we leave it to future work to improve this
# aspect of the model.
observation_distribution = tfd.Masked(
    tfd.MultivariateNormalDiag(loc=true_means, scale_diag=trainable_Sigma),
    tf.math.is_finite(observations[:, :1]))

def construct_pi(logit_pi_T):
  """Constructs initial distribution from logit of pi_T."""
  pi_T = sigmoid(logit_pi_T)
  tau_D0 = (1 - pi_T) / (num_states-1)
  pi = [pi_T]
  pi.extend([tau_D0 for _ in range(num_states-1)])
  return pi

def construct_Pi(logit_tau_T2, logit_tau_DT, logit_tau_D2):
  """Constructs transition matrix from logits of tau_T2, tau_DT, and tau_D2."""
  tau_T2 = sigmoid(logit_tau_T2)
  tau_DT = sigmoid(logit_tau_DT)
  tau_D2 = sigmoid(logit_tau_D2)
  tau_TD = (1 - sigmoid(logit_tau_T2)) / (num_states-1)
  tau_DD = (1 - (sigmoid(logit_tau_DT) + sigmoid(logit_tau_D2))) / (num_states-2)

  target_row = [tau_T2]
  target_row.extend([tau_TD for _ in range(num_states-1)])
  Pi = [target_row]
  for distractor_row_idx in range(1, num_states):
    new_row = [tau_DT]
    for distractor_col_idx in range(1, num_states):
      if distractor_col_idx == distractor_row_idx:
        new_row.append(tau_D2)
      else:
        new_row.append(tau_DD)
    Pi.append(new_row)

  return Pi

def construct_hmm(logit_pi_T, logit_tau_T2, logit_tau_DT, logit_tau_D2):
  """Constructs the HMM model from its free parameters."""
  pi = tfd.Categorical(probs=construct_pi(logit_pi_T))
  Pi = tfd.Categorical(probs=construct_Pi(logit_tau_T2, logit_tau_DT, logit_tau_D2))
  return tfd.HiddenMarkovModel(
      initial_distribution=pi,
      transition_distribution=Pi,
      time_varying_observation_distribution=True,
      observation_distribution=observation_distribution,
      num_steps=trial_len)

# SPECIFY LOG-LIKELIHOOD TRAINING OBJECTIVE AND OPTIMIZER
def log_prob():
  hmm = construct_hmm(trainable_logit_pi_T,
                      trainable_logit_tau_T2,
                      trainable_logit_tau_DT,
                      trainable_logit_tau_D2)
  return hmm.log_prob(observations)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

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

# RUN FORWARD-BACKWARD ALGORITHM TO COMPUTE MARGINAL POSTERIORS
def plot_posteriors():
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
    labs = [l.get_label() for l in ln1]
    ax.legend(ln1, labs, loc=4)
    ax.grid(True, color='white')
  
  # PLOT MARGINAL POSTERIORS PROBABILITIES OF EACH STATE
  fig = plt.figure(figsize=(10, 10))
  plot_state_posterior(fig.add_subplot(4, 2, 1),
                       posterior_probs[:, 0],
                       title="Target")
  for distractor_idx in range(1, num_states):
    plot_state_posterior(fig.add_subplot(4, 2, distractor_idx+1),
                         posterior_probs[:, distractor_idx],
                         title=f"Distractor {distractor_idx}")
  plt.subplot(4, 2, 8)
  plt.plot(np.isnan(observations[:, 0]))
  plt.ylabel('Missing')
  plt.tight_layout()
  plt.show()

# FIT MODEL
loss_history = []
for step in range(num_iters):
  loss, logit_pi_T, logit_tau_T2, logit_tau_DT, logit_tau_D2, Sigma = [t.numpy() for t in train_op()]

  # Format probabilities nicely for printing.
  pi = [tau.numpy() for tau in construct_pi(logit_pi_T)]
  Pi = np.array([[tau.numpy() for tau in row] for row in construct_Pi(logit_tau_T2, logit_tau_DT, logit_tau_D2)])

  loss_history.append(loss)
  if step % 500 == 0:
    print("step {}: log prob {} Sigma {}\npi\n{}\nPi\n{}".format(step, -loss, Sigma, pi, Pi))
    plot_posteriors()
print("Inferred pi:\n{}".format(pi))
print("Inferred Pi:\n{}".format(Pi))
print("Inferred Sigma:\n{}".format(tf.linalg.diag(Sigma)))

plt.figure()
plt.plot(loss_history)

plt.show()
