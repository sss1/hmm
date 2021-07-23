import numpy as np
from matplotlib import pylab as plt
import pandas as pd
import scipy.stats
import sys
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize, linewidth=144)

import load_subjects as ls
import plot_video

trials_to_include = range(1, 11)  # Omit practice trials
num_trials = len(trials_to_include)
num_iters = 3001  # Number of training iterations
num_objects = 7
num_modes = 3
num_states = num_objects + 2  # Add states for On-Task and Disengaged modes

def sigmoid(x):
  """Computes the sigmoid function."""
  return tf.math.sigmoid(x)

def logit(x):
  """Computes the logit function, i.e. the logistic sigmoid inverse."""
  return - tf.math.log(1. / x - 1.)

def train_model(subject):

  # Compute the lengths of the longest trial to allocate enough space for the data
  max_trial_len = 0
  for trial_idx, trial in enumerate(trials_to_include):
    max_trial_len = max(max_trial_len, subject.experiments['noshrinky'].datatypes['eyetrack'].trials[trial].data.shape[0])
  
  trial_lens = np.zeros((num_trials), dtype=int)
  observations = np.zeros((num_trials, max_trial_len, 2), dtype=np.float32)
  true_means = np.zeros((num_trials, max_trial_len, num_objects, 2), dtype=np.float32)
  
  for trial_idx, trial in enumerate(trials_to_include):
    trial_len = subject.experiments['noshrinky'].datatypes['eyetrack'].trials[trial].data.shape[0]
    trial_lens[trial_idx] = trial_len
    observations[trial_idx, :trial_len, :] = subject.experiments['noshrinky'].datatypes['eyetrack'].trials[trial].data[:, 1:]
    true_means[trial_idx, :trial_len, :, :] = subject.experiments['noshrinky'].datatypes['trackit'].trials[trial].object_positions.transpose(1, 0, 2)
  valid_data_mask = tf.math.logical_and(tf.math.is_finite(observations[:, :, 0]), tf.sequence_mask(trial_lens))

  # Dimensions of screen to be used for off-task distribution
  x_min = np.nanmin(observations[:, :, 0]) - 1.0
  x_max = np.nanmax(observations[:, :, 0]) + 1.0
  y_min = np.nanmin(observations[:, :, 1]) - 1.0
  y_max = np.nanmax(observations[:, :, 1]) + 1.0

  # CONSTRUCT HMM WITH TRAINABLE EMISSION DISTRIBUTION
  # pi is distribution of the initial object of attention.
  # We represent it in terms of logit-probabilities and then
  # take sigmoids+normalize after optimization, as this is much simpler than
  # constraining the probabilities themselves.
  trainable_logit_pi_D = tf.Variable(logit(1/num_modes), name='logit_pi_D', dtype='float32')
  trainable_logit_pi_O = tf.Variable(logit(1/num_modes), name='logit_pi_O', dtype='float32')
  
  # Pi is the transition matrix of the object of attention.
  # We represent it in terms of logits too.
  init_mode_switch_prob = 0.005
  init_object_switch_prob = 0.05
  trainable_logit_Pi_DO = tf.Variable(logit(init_mode_switch_prob), name='logit_Pi_DO', dtype='float32')
  trainable_logit_Pi_DI = tf.Variable(logit(init_mode_switch_prob), name='logit_Pi_DI', dtype='float32')
  trainable_logit_Pi_OD = tf.Variable(logit(init_mode_switch_prob), name='logit_Pi_OD', dtype='float32')
  trainable_logit_Pi_OI = tf.Variable(logit(init_mode_switch_prob), name='logit_Pi_OI', dtype='float32')
  trainable_logit_Pi_ID = tf.Variable(logit(init_mode_switch_prob), name='logit_Pi_ID', dtype='float32')
  trainable_logit_Pi_IO = tf.Variable(logit(init_mode_switch_prob), name='logit_Pi_IO', dtype='float32')
  trainable_logit_object_switch_prob = tf.Variable(logit(init_object_switch_prob), name='logit_object_switch_prob', dtype='float32')

  # Sigma is the (diagonal) covariance matrix of gaze around the attended object.
  # We assume that the x- and y-components of the variance (around the attended
  # object) are independent. In reality, this might not be the case, e.g., (if the
  # object is moving diagonally), but we leave it to future work to improve this
  # aspect of the model.
  Sigma_0 = 100.0
  trainable_Sigma = tf.Variable([Sigma_0, Sigma_0], name='Sigma')
 
  # CONSTRUCT EMISSION DISTRIBUTION (OF OBSERVATIONS GIVEN STATE)
  # Because of constraints in batching in TensorFlow Probability, we construct
  # this as a degenerate mixture of Gaussian and Uniform distributions, with
  # 100% weight on the Gaussian in the object states and 100% weight on the
  # uniform distribution in the off-task state.
  final_size = (num_trials, max_trial_len, num_states)
  categorical_distribution = tfd.Categorical(
      probs=np.broadcast_to(np.array(8*[[1.0, 0.0]] + [[0.0, 1.0]], dtype=np.float32),
                            shape=(final_size) + (2,)))
  # Distributions of gaze around attended objects. Note that there is a extra
  # row that will never be used because of the degenerate mixture weights.
  object_distributions = tfd.MultivariateNormalDiag(
      loc=np.concatenate((true_means, true_means[:, :, 0:2, :]), axis=2),
      scale_diag=trainable_Sigma)
  # Uniform distribution of gaze when not following an object.
  off_task_distribution = tfd.Independent(
      tfd.Uniform(low=np.array([x_min, y_min], dtype=np.float32),
                  high=np.array([x_max, y_max], dtype=np.float32)),
      reinterpreted_batch_ndims=1)
  observation_distribution = tfd.Mixture(
      cat=categorical_distribution,
      components=[object_distributions,
                  tfd.BatchBroadcast(off_task_distribution, to_shape=final_size)])
  observation_distribution = tfd.Masked(observation_distribution,
      validity_mask=tf.expand_dims(valid_data_mask, axis=2))

  def construct_pi(logit_pi_D, logit_pi_O):
    """Constructs initial distribution from logits of pi_D and pi_O."""
    pi_D = sigmoid(logit_pi_D)
    # Multiply by (1 - p_D) to ensure p_D + p_O <= 1
    pi_O = (1 - pi_D) * sigmoid(logit_pi_O)
    pi_I = 1 - (pi_D + pi_O)
    return num_objects * [pi_D/num_objects] + [pi_O, pi_I]
  
  def construct_Pi(logit_Pi_DO, logit_Pi_DI, logit_Pi_OD, logit_Pi_OI, logit_Pi_ID, logit_Pi_IO, logit_object_switch_prob):
    """Constructs transition matrix from logits of tau_T2, tau_DT, and tau_D2."""

    Pi_DO = sigmoid(trainable_logit_Pi_DO)
    Pi_DI = sigmoid(trainable_logit_Pi_DI)
    Pi_OD = sigmoid(trainable_logit_Pi_OD)
    Pi_OI = sigmoid(trainable_logit_Pi_OI)
    Pi_ID = sigmoid(trainable_logit_Pi_ID)
    Pi_IO = sigmoid(trainable_logit_Pi_IO)
    object_switch_prob = sigmoid(logit_object_switch_prob)
    object_stay_prob = 1 - (num_objects-1) * object_switch_prob
    Pi_DD = 1 - (Pi_DO + Pi_DI)
    Pi_OO = 1 - (Pi_OD + Pi_OI)
    Pi_II = 1 - (Pi_ID + Pi_IO)

    def get_entry(source_idx, dest_idx):
      if source_idx < num_objects:
        if dest_idx < num_objects:
          if source_idx == dest_idx:
            return Pi_DD * object_stay_prob
          else:
            return Pi_DD * object_switch_prob
        elif dest_idx == num_objects:
          return Pi_DO
        else:
          return Pi_DI
      elif source_idx == num_objects:
        if dest_idx < num_objects:
          return Pi_OD
        elif dest_idx == num_objects:
          return Pi_OO
        else:
          return Pi_OI
      else:
        if dest_idx < num_objects:
          return Pi_ID
        elif dest_idx == num_objects:
          return Pi_IO
        else:
          return Pi_II

    Pi = []
    for row_idx in range(num_states):
      new_row = []
      for col_idx in range(num_states):
        new_row.append(get_entry(row_idx, col_idx))
      Pi.append(new_row)

    return Pi

    # tau_T2 = sigmoid(logit_tau_T2)
    # tau_DT = sigmoid(logit_tau_DT)
    # tau_D2 = sigmoid(logit_tau_D2)
    # tau_TD = (1 - sigmoid(logit_tau_T2)) / (num_states-1)
    # tau_DD = (1 - (sigmoid(logit_tau_DT) + sigmoid(logit_tau_D2))) / (num_states-2)
  
    # target_row = [tau_T2]
    # target_row.extend([tau_TD for _ in range(num_states-1)])
    # Pi = [target_row]
    # for distractor_row_idx in range(1, num_states):
    #   new_row = [tau_DT]
    #   for distractor_col_idx in range(1, num_states):
    #     if distractor_col_idx == distractor_row_idx:
    #       new_row.append(tau_D2)
    #     else:
    #       new_row.append(tau_DD)
    #   Pi.append(new_row)
  
    # return Pi
  
  def construct_hmm(logit_pi_D, logit_pi_O, logit_Pi_DO, logit_Pi_DI, logit_Pi_OD, logit_Pi_OI, logit_Pi_ID, logit_Pi_IO, logit_object_switch_prob):
    """Constructs the HMM model from its free parameters."""
    pi = tfd.Categorical(probs=construct_pi(logit_pi_D, logit_pi_O))
    Pi = tfd.Categorical(probs=construct_Pi(logit_Pi_DO, logit_Pi_DI, logit_Pi_OD, logit_Pi_OI, logit_Pi_ID, logit_Pi_IO, logit_object_switch_prob))
    return tfd.HiddenMarkovModel(
        initial_distribution=pi,
        transition_distribution=Pi,
        time_varying_observation_distribution=True,
        observation_distribution=observation_distribution,
        num_steps=max_trial_len)
  
  # SPECIFY LOG-LIKELIHOOD TRAINING OBJECTIVE AND OPTIMIZER
  def log_prob():
    hmm = construct_hmm(trainable_logit_pi_D,
                        trainable_logit_pi_O,
                        trainable_logit_Pi_DO,
                        trainable_logit_Pi_DI,
                        trainable_logit_Pi_OD,
                        trainable_logit_Pi_OI,
                        trainable_logit_Pi_ID,
                        trainable_logit_Pi_IO,
                        trainable_logit_object_switch_prob)
    return tf.math.reduce_sum(hmm.log_prob(observations))
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
  
  @tf.function(autograph=True)
  def train_op():
    # Apply a gradient update.
    with tf.GradientTape() as tape:
      neg_log_prob = -log_prob()
    grads = tape.gradient(neg_log_prob, [trainable_logit_pi_D,
                                         trainable_logit_pi_O,
                                         trainable_logit_Pi_DO,
                                         trainable_logit_Pi_DI,
                                         trainable_logit_Pi_OD,
                                         trainable_logit_Pi_OI,
                                         trainable_logit_Pi_ID,
                                         trainable_logit_Pi_IO,
                                         trainable_logit_object_switch_prob,
                                         trainable_Sigma])
    optimizer.apply_gradients([(grads[0], trainable_logit_pi_D),
                               (grads[1], trainable_logit_pi_O),
                               (grads[2], trainable_logit_Pi_DO),
                               (grads[3], trainable_logit_Pi_DI),
                               (grads[4], trainable_logit_Pi_OD),
                               (grads[5], trainable_logit_Pi_OI),
                               (grads[6], trainable_logit_Pi_ID),
                               (grads[7], trainable_logit_Pi_IO),
                               (grads[8], trainable_logit_object_switch_prob),
                               (grads[9], trainable_Sigma)])
  
    return neg_log_prob, trainable_logit_pi_D, trainable_logit_pi_O, \
        trainable_logit_Pi_DO, trainable_logit_Pi_DI, trainable_logit_Pi_OD, \
        trainable_logit_Pi_OI, trainable_logit_Pi_ID, trainable_logit_Pi_IO, \
        trainable_logit_object_switch_prob, trainable_Sigma
  
  def plot_posteriors():
    # RUN FORWARD-BACKWARD ALGORITHM TO COMPUTE MARGINAL POSTERIORS
    hmm = construct_hmm(trainable_logit_pi_D,
                        trainable_logit_pi_O,
                        trainable_logit_Pi_DO,
                        trainable_logit_Pi_DI,
                        trainable_logit_Pi_OD,
                        trainable_logit_Pi_OI,
                        trainable_logit_Pi_ID,
                        trainable_logit_Pi_IO,
                        trainable_logit_object_switch_prob)
    posterior_dists = hmm.posterior_marginals(observations)
    posterior_probs = posterior_dists.probs_parameter().numpy()
    posterior_mode = hmm.posterior_mode(observations=observations)
    
    # def plot_state_posterior(ax, state_posterior_probs, title, legend=False):
    #   ln1 = ax.plot(state_posterior_probs, c='blue', lw=3, label='p(state | observations)')
    #   ax.set_ylim(0., 1.1)
    #   # ax.set_ylabel('posterior probability')
    #   labs = [l.get_label() for l in ln1]
    #   if legend:
    #     ax.legend(ln1, labs, loc=4)
    #   ax.grid(True, color='white')
    
    for trial in range(num_trials):
      plot_video.plot_trial_video(
          observations[trial, :trial_lens[trial], :],
          true_means[trial, :trial_lens[trial], :, :],
          posterior_mode[trial, :trial_lens[trial]])
      # # PLOT MARGINAL POSTERIORS PROBABILITIES OF EACH STATE
      # fig = plt.figure(figsize=(10, 10))
      # plot_state_posterior(fig.add_subplot(8, 1, 1),
      #                      posterior_probs[trial, :, 0],
      #                      title="Target",
      #                      legend=True)
      # for distractor_idx in range(1, num_states):
      #   plot_state_posterior(fig.add_subplot(8, 1, distractor_idx+1),
      #                        posterior_probs[trial, :, distractor_idx],
      #                        title=f"Distractor {distractor_idx}")
      # plt.subplot(8, 1, 8)
      # plt.plot(tf.math.logical_not(valid_data_mask[trial, :]))
      # plt.ylim((0., 1.1))
      # plt.ylabel('Missing')
      # plt.tight_layout()
  
  # FIT MODEL
  loss_history = []
  pi_history = []
  Pi_history = []
  Sigma_history = []
  for step in tqdm(range(num_iters)):
    loss, logit_pi_D, logit_pi_O, logit_Pi_DO, logit_Pi_DI, logit_Pi_OD, \
        logit_Pi_OI, logit_Pi_ID, logit_Pi_IO, logit_object_switch_prob, \
        Sigma = [t.numpy() for t in train_op()]

    # Format probabilities nicely for printing.
    pi = np.array([tau.numpy() for tau in construct_pi(logit_pi_D, logit_pi_O)])
    Pi = np.array([[tau.numpy() for tau in row] for row in construct_Pi(logit_Pi_DO, logit_Pi_DI, logit_Pi_OD, logit_Pi_OI, logit_Pi_ID, logit_Pi_IO, logit_object_switch_prob)])
  
    if np.any(pi < 0):
      print(pi)
      raise ValueError('pi has negative entries!')
    if np.any(Pi < 0):
      print(Pi)
      raise ValueError('Pi has negative entries!')
    if np.isnan(loss):
      raise ValueError('loss has become NaN!')
  
    loss_history.append(loss)
    pi_history.append(pi)
    Pi_history.append(Pi)
    Sigma_history.append(Sigma)
    # if step % 10 == 0:
    #   print(f'Step {step}')
    # print("step {}: log prob {} Sigma {}\npi\n{}\nPi\n{}".format(step, -loss, Sigma, pi, Pi))
  print(f"Inferred pi:\n{pi}")
  print(f"Inferred Pi:\n{Pi}")
  print(f"Inferred Sigma:\n{tf.linalg.diag(Sigma)}")
  
  # plt.figure()
  # plt.subplot(8, 2, 1)
  # plt.plot(loss_history)
  # plt.ylabel('loss')

  # plt.subplot(8, 2, 2)
  # plt.plot([y[0] for y in pi_history])
  # plt.ylabel('pi_D')
  # plt.subplot(8, 2, 3)
  # plt.plot([y[7] for y in pi_history])
  # plt.ylabel('pi_O')
  # plt.subplot(8, 2, 4)
  # plt.plot([y[8] for y in pi_history])
  # plt.ylabel('pi_I')

  # plt.subplot(8, 2, 5)
  # plt.plot([y[0, 0] for y in Pi_history])
  # plt.ylabel('Pi_DD_stay')
  # plt.subplot(8, 2, 6)
  # plt.plot([y[0, 1] for y in Pi_history])
  # plt.ylabel('Pi_DD_switch')
  # plt.subplot(8, 2, 7)
  # plt.plot([y[0, 7] for y in Pi_history])
  # plt.ylabel('Pi_DO')
  # plt.subplot(8, 2, 8)
  # plt.plot([y[0, 8] for y in Pi_history])
  # plt.ylabel('Pi_DI')
  # plt.subplot(8, 2, 9)
  # plt.plot([y[7, 0] for y in Pi_history])
  # plt.ylabel('Pi_OD')
  # plt.subplot(8, 2, 10)
  # plt.plot([y[7, 7] for y in Pi_history])
  # plt.ylabel('Pi_OO')
  # plt.subplot(8, 2, 11)
  # plt.plot([y[7, 8] for y in Pi_history])
  # plt.ylabel('Pi_OI')
  # plt.subplot(8, 2, 12)
  # plt.plot([y[8, 0] for y in Pi_history])
  # plt.ylabel('Pi_ID')
  # plt.subplot(8, 2, 13)
  # plt.plot([y[8, 7] for y in Pi_history])
  # plt.ylabel('Pi_IO')
  # plt.subplot(8, 2, 14)
  # plt.plot([y[8, 8] for y in Pi_history])
  # plt.ylabel('Pi_II')

  # plt.subplot(8, 2, 15)
  # plt.plot([y[0] for y in Sigma_history])
  # plt.ylabel('Sigma_x')
  # plt.subplot(8, 2, 16)
  # plt.plot([y[1] for y in Sigma_history])
  # plt.ylabel('Sigma_y')
  # plot_posteriors()
  # plt.show()
  return loss, pi, Pi, Sigma

# LOAD DATA
subjects = ls.load_all_data()
num_subjects = len(subjects)

table_as_dict = {'subject_ID': [], 'age': [], 'loss': [], 'pi_D': [], 'pi_O': [], 'pi_I': [],
                 'Pi_DD_stay': [], 'Pi_DD_switch': [], 'Pi_DO': [], 'Pi_DI': [], 'Pi_OD': [],
                 'Pi_OO': [], 'Pi_OI': [], 'Pi_ID': [], 'Pi_IO': [],
                 'Pi_II': [], 'Sigma_x': [], 'Sigma_y': []}
for subject in subjects.values():
  table_as_dict['subject_ID'].append(subject.ID)
  table_as_dict['age'].append(float(subject.experiments['noshrinky'].datatypes['trackit'].metadata['Age']))
  print(f'Training model on subject {subject.ID}...')
  loss, pi, Pi, Sigma = train_model(subject)

  table_as_dict['loss'].append(loss)
  table_as_dict['pi_D'].append(pi[0])
  table_as_dict['pi_O'].append(pi[7])
  table_as_dict['pi_I'].append(pi[8])

  table_as_dict['Pi_DD_stay'].append(Pi[0, 0])
  table_as_dict['Pi_DD_switch'].append(Pi[0, 1])
  table_as_dict['Pi_DO'].append(Pi[0, 7])
  table_as_dict['Pi_DI'].append(Pi[0, 8])

  table_as_dict['Pi_OD'].append(Pi[7, 0])
  table_as_dict['Pi_OO'].append(Pi[7, 7])
  table_as_dict['Pi_OI'].append(Pi[7, 8])

  table_as_dict['Pi_ID'].append(Pi[8, 0])
  table_as_dict['Pi_IO'].append(Pi[8, 7])
  table_as_dict['Pi_II'].append(Pi[8, 8])

  table_as_dict['Sigma_x'].append(Sigma[0])
  table_as_dict['Sigma_y'].append(Sigma[1])

df = pd.DataFrame(table_as_dict)
df.to_csv('training_results_by_age_hhmm.csv', index=False)
