import math
import numpy as np
from itertools import chain
import tensorflow as tf

import load_subjects as ls

# Given a list X, returns a list of changepoints
def get_changepoints(X):
  return X[:-1] != X[1:]

# Construct numpy array from jagged data by filling ends of short rows with NaNs
def jagged_to_numpy(jagged):
  aligned = np.ones((len(jagged), max([len(row) for row in jagged]))) * np.nan # allocate appropriately sized array of NaNs
  for i, row in enumerate(jagged): # populate columns
    aligned[i, :len(row)] = row
  return aligned

def __interpolate_to_length_labels(X, N):
  if not N == int(N):
    raise ValueError('New length must be an integer, but is ' + str(N))
  N = int(N)
  change_points = np.where(X[:-1] != X[1:])[0]
  X_new = np.zeros(N, dtype = int)
  upsample_rate = float(N) / len(X)
  new_segment_end = 0 # need this for the edge case where there are no change points
  for change_point_idx in range(len(change_points)):
    change_point = change_points[change_point_idx] + 1
    if change_point_idx == 0:
      prev_change_point = 0
    new_segment_start = int(math.ceil(prev_change_point * upsample_rate))
    new_segment_end = int(math.ceil(change_point * upsample_rate))
    X_new[new_segment_start:new_segment_end] = X[prev_change_point]
    X_new[new_segment_start:new_segment_end] = X[prev_change_point]
    prev_change_point = change_point
  X_new[new_segment_end:] = X[-1] # manually fill-in after last change point
  return X_new

# X = np.array([0, 0, 0, 0, 1, 1, 0, 1])
# print X
# print __interpolate_to_length_labels(X, len(X))
# print __interpolate_to_length_labels(X, 2*len(X))
# print __interpolate_to_length_labels(X, 2.5*len(X))

# Given a K x D x N array of numbers, encoding the positions of each of K D-dimensional objects over N time points,
# performs interpolate_to_length_D (independently) on each object in X
def interpolate_to_length_distractors(X, new_len):
  K = X.shape[0]
  D = X.shape[1]
  X_new = np.zeros((K, D, new_len))
  for k in range(K):
    X_new[k,:,:] = interpolate_to_length_D(X[k,:,:], new_len)
  return X_new

# Given a D-dimensional sequence X of numbers, performs interpolate_to_length (independently) on each dimension of X
# X is D x N, where D is the dimensionality and N is the sample length
def interpolate_to_length_D(X, new_len):
  D = X.shape[0]
  X_new = np.zeros((D, new_len))
  for d in range(D):
    X_new[d, :] = __interpolate_to_length(X[d, :], new_len)
  return X_new

# Given a sequence X of numbers, returns the length-new_len linear interpolant of X
def __interpolate_to_length(X, new_len):
  old_len = X.shape[0]
  return np.interp([(float(n)*old_len)/new_len for n in range(new_len)], range(old_len), X)

# Given a D-dimensional sequence X of numbers, performs impute_missing_data_D (independently) on each dimension of X
# X is D x N, where D is the dimensionality and N is the sample length
def impute_missing_data_D(X, max_len = 10):
  D = X.shape[0]
  for d in range(D):
    X[d, :] = __impute_missing_data(X[d, :], max_len)
  return X

# Given a sequence X of floats, replaces short streches (up to length max_len) of NaNs with linear interpolation
# For example, if
# X = np.array([1, NaN, NaN,  4, NaN,  6])
# then
# impute_missing_data(X, max_len = 1) == np.array([1, NaN, NaN, 5, 6])
# and
# impute_missing_data(X, max_len = 2) == np.array([1, 2, 3, 4, 5, 6])
def __impute_missing_data(X, max_len):
  last_valid_idx = -1
  for n in range(len(X)):
    if not math.isnan(X[n]):
      if last_valid_idx < n - 1: # there is missing data and we have seen at least one valid eyetracking sample
        if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
          if last_valid_idx == -1: # No previous valid data (i.e., first timepoint is missing)
            X[0:n] = X[n] # Just propogate first valid data point
          else:
            first_last = np.array([X[last_valid_idx], X[n]]) # initial and final values from which to linearly interpolate
            new_len = n - last_valid_idx + 1
            X[last_valid_idx:(n + 1)] = np.interp([float(x)/(new_len - 1) for x in range(new_len)], [0, 1], first_last)
      last_valid_idx = n
    elif n == len(X) - 1: # if n is the last index of X and X[n] is NaN
      if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
        X[last_valid_idx:] = X[last_valid_idx]
  return X

# Some test cases:
# X = np.array([1, 2, 3])
# print __impute_missing_data(X, 10)
# X = np.array([0, 1])
# print __interpolate_fixed_length(X, 5)
# X = np.array([1, float('nan'), float('nan'), float('nan'), 5, float('nan'), float('nan'), 8])
# print X
# print __impute_missing_data(X, 3)

# Given two binary sequences xs and ys of equal length, computes a confusion between xs and ys,
# but allows for some slack in detecting positives (i.e., positives with distance max_dist can be counted correct)
def generalized_2x2_confusion_matrix(xs, ys, max_dist):
  # y_positives version of ys with 1s propogated max_dist on either side of original 1s
  # y_positives is compared with xs to compute true positives (TPs). Rest of confusion matrix can be calculated just from TPs, xs, and ys.
  y_positives = np.zeros(np.shape(ys), dtype=bool)
  for offset in range(-max_dist,max_dist+1):
    last_idx1 = None if offset >= 0 else offset # have to use None to index through end of array
    last_idx2 = None if offset <= 0 else -offset
    y_positives[max(0,offset):last_idx1] = np.logical_or(y_positives[max(0, offset):last_idx1],
                                                                  ys[max(0,-offset):last_idx2])
  TPs = np.sum(np.logical_and(xs, y_positives))
  FPs = np.sum(xs) - TPs
  FNs = max(0, np.sum(ys) - TPs)
  TNs = len(xs) - (TPs + FPs + FNs)

  return np.array([[TNs, FNs],[FPs, TPs]])

# A simple test:
# Compare to reference confusion matrix implementation for max_dist=0,
# And check that classification performance improves with larger max_dist
# >>> from sklearn.metrics import confusion_matrix
# >>> xs = np.random.randint(low=0,high=2,size=10000)
# >>> ys = np.random.randint(low=0,high=2,size=10000)
# >>> confusion_matrix(xs, ys, labels=[False, True])
# array([[2479, 2531],
#        [2455, 2535]])
# >>> util.generalized_2x2_confusion_matrix(xs, ys, max_dist=0)
# array([[2479, 2531],
#        [2455, 2535]])
# >>> util.generalized_2x2_confusion_matrix(xs, ys, max_dist=1)
# array([[4337,  673],
#        [ 597, 4393]])
# >>> util.generalized_2x2_confusion_matrix(xs, ys, max_dist=1000)
# array([[4934,   76],
#        [   0, 4990]])

# Given a confusion matrix CM from a binary classification task; CM should be formatted as
# [[True Negatives, False Negatives],
#  [False Positives, True Positives]]
def classification_performance(CM):
  precision = float(CM[1,1])/(CM[1,1] + CM[1,0])
  recall = float(CM[1,1])/(CM[1,1] + CM[0,1])
  F1 = 2.0 * precision * recall / (precision + recall)
  MCC = (float(CM[0,0])*CM[1,1] - float(CM[0,1])*CM[1,0])/math.sqrt((CM[1,1] + CM[1,0])*(CM[1,1]+CM[0,1])*(CM[0,0]+CM[1,0])*(CM[0,0]+CM[0,1]))
  return precision, recall, F1, MCC

# Interpolate missing eyetracking data and store new imputed data proportion
def impute_missing_data(experiment, max_len = 10):
  if not 'eyetrack' in experiment.datatypes:
    return # If eyetracking data is missing, nothing to do
  eyetrack = experiment.datatypes['eyetrack']
  impute_missing_data_D(eyetrack.raw_data.T, max_len=max_len).T
  eyetrack.proportion_missing_frames_after_imputation = np.mean(np.isnan(eyetrack.raw_data[:, 1]))
  eyetrack.proportion_imputed_frames = eyetrack.proportion_missing_frames_after_imputation - eyetrack.proportion_total_missing_frames

# Break experiment's eyetracking data into trialsprint('After: ' + str(trackit_trial.object_positions.shape))
def break_eyetracking_into_trials(experiment):
  if not 'trackit' in experiment.datatypes or not 'eyetrack' in experiment.datatypes:
    experiment.has_all_experiment_data = False
    return # If either TrackIt or eyetracking data is missing, nothing to do
  experiment.has_all_experiment_data = True
  trackit, eyetrack = experiment.datatypes['trackit'], experiment.datatypes['eyetrack']
  eyetrack.trials = []
  for (trial_idx, trial) in enumerate(trackit.trials):
    trial_start, trial_end = trial.timestamps[0], trial.timestamps[-1]
    trial_eyetrack_data = np.asarray([frame for frame in eyetrack.raw_data if trial_start < frame[0] and frame[0] < trial_end])
    eyetrack.trials.append(ls.Eyetrack_Trial_Data(trial_eyetrack_data))
    eyetrack.trials[-1].proportion_missing_frames = np.mean(np.isnan(trial_eyetrack_data[:,1]))

# Interpolate the TrackIt data points to be synchronized with the Eyetracking data
def interpolate_trackit_to_eyetracking(experiment):
  if not experiment.has_all_experiment_data:
    return  # If either TrackIt or eyetracking data is missing, nothing to do
  trackit, eyetrack = experiment.datatypes['trackit'], experiment.datatypes['eyetrack']
  for (trackit_trial, eyetrack_trial) in zip(trackit.trials, eyetrack.trials):
    eyetrack_len = eyetrack_trial.data.shape[0]
    interpolated_object_positions = np.zeros((trackit_trial.object_positions.shape[0], eyetrack_len, 2))
    # X coordinates
    interpolated_object_positions[:, :, 0] = interpolate_to_length_D(trackit_trial.object_positions[:, :, 0], new_len=eyetrack_len)
    # Y coordinates
    interpolated_object_positions[:, :, 1] = interpolate_to_length_D(trackit_trial.object_positions[:, :, 1], new_len=eyetrack_len)
    trackit_trial.object_positions = interpolated_object_positions

# Annotates experiment with the trials to be filtered, as well as whether the entire experiment should be filtered.
# Also excludes practice trials.
def filter_experiment(experiment, min_prop_data_per_trial=0.5, min_prop_trials_per_subject=0.5):
  try:
    eyetrack = experiment.datatypes['eyetrack']
  except KeyError: # If the experiment doesn't have eyetracking data, nothing to do
    return
  except AttributeError as e:
    print("AttributeError: " + str(e))
    print('Perhaps, the eyetracking data has not yet been broken into trials. Run break_eyetracking_into_trials(experiment) first.')
    return
  trials = eyetrack.trials
  experiment.trials_to_keep = [idx for (idx, trial) in enumerate(trials) \
                                  if 1 - trial.proportion_missing >= min_prop_data_per_trial and idx > 0]
  experiment.keep_experiment = (len(experiment.trials_to_keep) >= len(trials) * min_prop_trials_per_subject)


def logit(x):
  """Computes the logit function, i.e. the logistic sigmoid inverse."""
  return - tf.math.log(1. / x - 1.)


def format_model_args_from_row(subject_ID, model_args_df, condition):

  row_condition = (model_args_df['subject_ID'] == int(subject_ID)) & (model_args_df['condition'] == condition)
  model_args_names = ['pi_D', 'pi_O', 'Pi_DO', 'Pi_DI', 'Pi_OD', 'Pi_OI',
                      'Pi_ID', 'Pi_IO', 'object_switch_prob', 'Sigma_x',
                      'Sigma_y']
  model_args_row = model_args_df[row_condition][model_args_names].values[0]

  # Probability parameters need to be passed as logits
  model_args = [np.array(logit(x).numpy(), dtype=np.float32) for x in model_args_row[0:9]]
  # Sigma parameter needs to be passed as length-2 vector
  model_args += [np.array([model_args_row[9], model_args_row[10]], dtype=np.float32)]
  return model_args
