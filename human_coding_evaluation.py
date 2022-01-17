import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hhmm
import load_subjects as ls
import plot_video
import util

subjects = ls.load_all_data()

np.set_printoptions(suppress=True)

sessions_to_use = [('coder1', '0',  'noshrinky'),
                   ('coder1', '0',  'shrinky'  ),
                   ('coder1', '31', 'noshrinky'),
                   ('coder1', '31', 'shrinky'  ),
                   ('coder1', '7',  'noshrinky'),
                   ('coder1', '7',  'shrinky'  ),
                   ('coder1', '27', 'noshrinky'),
                   ('coder1', '15', 'shrinky'  ),
                   ('coder1', '28', 'noshrinky'),
                   ('coder1', '28', 'shrinky'  ),
                   ('coder1', '47', 'noshrinky'),
                   ('coder1', '46', 'noshrinky'),
                   ('coder1', '46', 'shrinky'  ),
                   ('coder1', '26', 'noshrinky'),
                   ('coder1', '26', 'shrinky'  ),
                   ('coder1', '30', 'noshrinky'),
                   ('coder1', '18', 'noshrinky'),
                   ('coder1', '18', 'shrinky'  ),
                   ('coder1', '35', 'shrinky'  ),
                   ('coder1', '19', 'noshrinky'),
                   ('coder1', '19', 'shrinky'  ),
                   ('coder1', '3',  'noshrinky'),
                   ('coder1', '3',  'shrinky'  ),
                   ('coder1', '48', 'noshrinky'),
                   ('coder1', '48', 'shrinky'  ),
                   ('coder1', '45', 'noshrinky'),
                   ('coder1', '45', 'shrinky'  ),
                   ('coder1', '1',  'noshrinky'),
                   ('coder1', '6',  'noshrinky'),
                   ('coder1', '41', 'noshrinky'),
                   ('coder1', '41', 'shrinky'  ),
                   ('coder1', '10', 'noshrinky'),
                   ('coder1', '10', 'shrinky'  ),
                   ('coder1', '9',  'noshrinky'),
                   ('coder1', '9',  'shrinky'  ),
                   ('coder1', '38', 'noshrinky'),
                   ('coder1', '38', 'shrinky'  ),
                   ('coder1', '25', 'noshrinky'),
                   ('coder1', '25', 'shrinky'  ),
                   ('coder1', '21', 'noshrinky'),
                   ('coder1', '21', 'shrinky'  ),
                   ('coder1', '20', 'noshrinky'),
                   ('coder1', '33', 'noshrinky'),
                   ('coder1', '33', 'shrinky'  ),
                   ('coder1', '11', 'noshrinky'),
                   ('coder1', '11', 'shrinky'  ),
                   ('coder1', '49', 'noshrinky'),
                   ('coder1', '49', 'shrinky'  ),
                   ('coder1', '13', 'noshrinky'),
                   ('coder1', '13', 'shrinky'  ),
                   ('coder1', '32', 'noshrinky'),
                   ('coder1', '32', 'shrinky'  ),
                   ('coder1', '24', 'shrinky'  ),
                   ('coder1', '2',  'noshrinky'),
                   ('coder1', '37', 'shrinky'  ),
                   ('coder1', '17', 'noshrinky'),
                   ('coder1', '17', 'shrinky'  ),
                   ('coder1', '22', 'noshrinky'),
                   ('coder1', '22', 'shrinky'  ),
                   ('coder3', '0',  'shrinky'  ),
                   ('coder3', '7',  'shrinky'  ),
                   ('coder3', '27', 'noshrinky'),
                   ('coder3', '42', 'noshrinky'),
                   ('coder3', '15', 'noshrinky'),
                   ('coder3', '15', 'shrinky'  ),
                   ('coder3', '28', 'shrinky'  ),
                   ('coder3', '47', 'shrinky'  ),
                   ('coder3', '35', 'noshrinky'),
                   ('coder3', '19', 'noshrinky'),
                   ('coder3', '3',  'noshrinky'),
                   ('coder3', '48', 'noshrinky'),
                   ('coder3', '14', 'noshrinky'),
                   ('coder3', '36', 'noshrinky'),
                   ('coder3', '36', 'shrinky'  ),
                   ('coder3', '29', 'noshrinky'),
                   ('coder3', '29', 'shrinky'  ),
                   ('coder3', '4',  'noshrinky'),
                   ('coder3', '4',  'shrinky'  ),
                   ('coder3', '1',  'shrinky'  ),
                   ('coder3', '6',  'shrinky'  ),
                   ('coder3', '41', 'shrinky'  ),
                   ('coder3', '10', 'noshrinky'),
                   ('coder3', '9',  'noshrinky'),
                   ('coder3', '38', 'shrinky'  ),
                   ('coder3', '20', 'shrinky'  ),
                   ('coder3', '11', 'shrinky'  ),
                   ('coder3', '16', 'noshrinky'),
                   ('coder3', '16', 'shrinky'  ),
                   ('coder3', '32', 'shrinky'  ),
                   ('coder3', '24', 'noshrinky'),
                   ('coder3', '2',  'noshrinky'),
                   ('coder3', '2',  'shrinky'  ),
                   ('coder3', '37', 'noshrinky'),
                   ('coder3', '34', 'noshrinky'),
                   ('coder3', '34', 'shrinky'  ),
                   ('coder3', '17', 'noshrinky'),
                   ('coder3', '17', 'shrinky'  ),
                   ('coder3', '22', 'noshrinky'),
]

# Temporal slack for computing confusion matrices. Higher gives more lenient classification results.
# The paper presents results for num_slack_frames = 0 and num_slack_frames = 1.
num_slack_frames = 0

# For now, only include noshrinky sessions.
# TODO: Run HHMM on shrinky sessions and include these as well.
sessions_to_use = [session for session in sessions_to_use if session[2] == 'noshrinky']

model_args_df = pd.read_csv('training_results_by_age_hhmm_noOE.csv')
# object_switch_prob wasn't exported, so calculate this
model_args_df['Pi_DD'] = 1 - (model_args_df['Pi_DO'] + model_args_df['Pi_DI'])
model_args_df['object_switch_prob'] = model_args_df['Pi_DD_switch']/model_args_df['Pi_DD']

model_subjects = set(model_args_df['subject_ID'])
human_coded_subjects = set([int(x[1]) for x in sessions_to_use])

HHMM_prop_correct_by_session = np.zeros((len(sessions_to_use),))
HHMM_precision_switches_by_session = np.zeros((len(sessions_to_use),))
HHMM_recall_switches_by_session = np.zeros((len(sessions_to_use),))
HHMM_F1_switches_by_session = np.zeros((len(sessions_to_use),))
HHMM_MCC_switches_by_session = np.zeros((len(sessions_to_use),))
total_confusion_matrix_switches = np.zeros((2,2))
HHMM_precision_off_task_by_session = np.zeros((len(sessions_to_use),))
HHMM_recall_off_task_by_session = np.zeros((len(sessions_to_use),))
HHMM_F1_off_task_by_session = np.zeros((len(sessions_to_use),))
HHMM_MCC_off_task_by_session = np.zeros((len(sessions_to_use),))
total_confusion_matrix_off_task = np.zeros((2,2))
for (session_idx, (coder, subject_ID, experiment)) in enumerate(sessions_to_use):

  true_means, trial_lens, observations = hhmm.format_data(subjects[subject_ID].experiments[experiment])
  valid_data_mask = hhmm.get_valid_data_mask(trial_lens, observations)

  # row_condition = (model_args_df['subject_ID'] == int(subject_ID)) & (model_args_df['condition'] == experiment)
  # model_args_row = model_args_df[row_condition][hhmm.model_args_names].values[0]
  # # Probability parameters need to be passed as logits
  # model_args = [np.array(hhmm.logit(x).numpy(), dtype=np.float32) for x in model_args_row[0:9]]
  # # Sigma parameter needs to be passed as length-2 vector
  # model_args += [np.array([model_args_row[9], model_args_row[10]], dtype=np.float32)]

  model_args = util.format_model_args_from_row(subject_ID, model_args_df, experiment)

  HHMM_MLE = hhmm.get_MLE_states(true_means, trial_lens, observations, model_args).numpy()
  HHMM_MLE[HHMM_MLE == 7] = 0  # Replace On-Task state with Target state

  session_HHMM_prop_correct_by_frame = []
  confusion_matrix_switches = np.zeros((2,2))
  confusion_matrix_off_task = np.zeros((2,2))

  session_changepoints_HHMM = []
  session_changepoints_human = []
  for trial in range(1, hhmm.num_trials):

    # Extract HHMM maximum likelihood estimate, subsampled by a factor of 6, to compare with human coding
    # plot_video.plot_trial_video(
    #     observations[trial, :trial_lens[trial], :],
    #     true_means[trial, :trial_lens[trial], :, :],
    #     HHMM_MLE[trial, :trial_lens[trial]])
    trial_HHMM = HHMM_MLE[trial, :trial_lens[trial]]
    trial_HHMM[~valid_data_mask[trial, :trial_lens[trial]]] = -1  # Code missing data as -1
    trial_HHMM = trial_HHMM[::6]

    human_coding_filename = f'human_coded/{coder}/{subject_ID}_{experiment}_trial_{trial+1}_coding.csv'
    trial_human = []
    with open(human_coding_filename, 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader, None) # Skip CSV header line
      for line in reader:
        if line[1] in ['Off Screen', '']: # Ignore these frames, since they don't correspond to decodable states
          trial_human.append(-1)
        elif line[1] == 'Off Task': # Ignore these frames, since they don't correspond to decodable states
          trial_human.append(8)
        else:
          trial_human.append(int(line[1][-1:])) # NOTE: This breaks if there are >10 total objects!
    trial_human = np.array(trial_human)

    # HHMM and human codings should always be within 2 frames of each other.
    if abs(trial_HHMM.shape[0] - trial_human.shape[0]) > 2:
      print(f'Coder: {coder} Subject: {subject_ID} Trial: {trial+1} len(HHMM): {trial_HHMM.shape[0]} len(human): {trial_human.shape[0]}')

    # Truncate to the shorter of the HHMM and human coding lengths
    if trial_HHMM.shape[0] < trial_human.shape[0]:
      trial_human = trial_human[:trial_HHMM.shape[0]]
    elif trial_HHMM.shape[0] > trial_human.shape[0]:
      trial_HHMM = trial_HHMM[:trial_human.shape[0]]

    # Remove frames where either coding is `Off Screen'
    to_keep = ((trial_human >= 0) & (trial_HHMM >= 0))
    trial_HHMM = trial_HHMM[to_keep]
    trial_human = trial_human[to_keep]

    # Extract off-task classifications
    HHMM_off_task = (trial_HHMM == 8)
    human_off_task = (trial_human == 8)

    # Remove frames where either coding is `Off Screen'
    to_keep = ((trial_human < 8) & (trial_HHMM < 8))
    trial_HHMM = trial_HHMM[to_keep]
    trial_human = trial_human[to_keep]

    # Concatenate accuracies for current trial
    session_HHMM_prop_correct_by_frame.extend(trial_HHMM == trial_human)

    # Add confusion matrix changepoints for current trial
    if len(trial_HHMM) > 0:
      changepoints_HHMM = util.get_changepoints(trial_HHMM)
      changepoints_human = util.get_changepoints(trial_human)
      confusion_matrix_switches += util.generalized_2x2_confusion_matrix(changepoints_HHMM, changepoints_human, max_dist=num_slack_frames)
      confusion_matrix_off_task += util.generalized_2x2_confusion_matrix(HHMM_off_task, human_off_task, max_dist=0)

    total_confusion_matrix_switches += confusion_matrix_switches
    total_confusion_matrix_off_task += confusion_matrix_off_task

  # Aggregate over frames or trials within session
  HHMM_prop_correct_by_session[session_idx] = np.mean(session_HHMM_prop_correct_by_frame)
  HHMM_precision_switches_by_session[session_idx], HHMM_recall_switches_by_session[session_idx], \
      HHMM_F1_switches_by_session[session_idx], HHMM_MCC_switches_by_session[session_idx] = \
      util.classification_performance(confusion_matrix_switches)
  HHMM_precision_off_task_by_session[session_idx], HHMM_recall_off_task_by_session[session_idx], \
      HHMM_F1_off_task_by_session[session_idx], HHMM_MCC_off_task_by_session[session_idx] = \
      util.classification_performance(confusion_matrix_off_task)

# Aggregate over sessions
HHMM_acc_means = np.nanmean(HHMM_prop_correct_by_session)
HHMM_acc_CIs = 1.96*np.nanstd(HHMM_prop_correct_by_session)/math.sqrt(len(sessions_to_use))
HHMM_switches_prec_means = np.nanmean(HHMM_precision_switches_by_session)
HHMM_switches_prec_CIs = 1.96*np.nanstd(HHMM_precision_switches_by_session)/math.sqrt(len(sessions_to_use))
HHMM_switches_rec_means = np.nanmean(HHMM_recall_switches_by_session)
HHMM_switches_rec_CIs = 1.96*np.nanstd(HHMM_recall_switches_by_session)/math.sqrt(len(sessions_to_use))
HHMM_switches_F1_means = np.nanmean(HHMM_F1_switches_by_session)
HHMM_switches_F1_CIs = 1.96*np.nanstd(HHMM_F1_switches_by_session)/math.sqrt(len(sessions_to_use))
HHMM_switches_MCC_means = np.nanmean(HHMM_MCC_switches_by_session)
HHMM_switches_MCC_CIs = 1.96*np.nanstd(HHMM_MCC_switches_by_session)/math.sqrt(len(sessions_to_use))
HHMM_off_task_prec_means = np.nanmean(HHMM_precision_off_task_by_session)
HHMM_off_task_prec_CIs = 1.96*np.nanstd(HHMM_precision_off_task_by_session)/math.sqrt(len(sessions_to_use))
HHMM_off_task_rec_means = np.nanmean(HHMM_recall_off_task_by_session)
HHMM_off_task_rec_CIs = 1.96*np.nanstd(HHMM_recall_off_task_by_session)/math.sqrt(len(sessions_to_use))
HHMM_off_task_F1_means = np.nanmean(HHMM_F1_off_task_by_session)
HHMM_off_task_F1_CIs = 1.96*np.nanstd(HHMM_F1_off_task_by_session)/math.sqrt(len(sessions_to_use))
HHMM_off_task_MCC_means = np.nanmean(HHMM_MCC_off_task_by_session)
HHMM_off_task_MCC_CIs = 1.96*np.nanstd(HHMM_MCC_off_task_by_session)/math.sqrt(len(sessions_to_use))


print(f'HHMM Accuracy: {HHMM_acc_means}  95% CI: ({HHMM_acc_means-HHMM_acc_CIs}, {HHMM_acc_means+HHMM_acc_CIs})')

print('HHMM\n', total_confusion_matrix_switches)
print(f'HHMM Switches Precision: {HHMM_switches_prec_means}  95% CI: ({HHMM_switches_prec_means-HHMM_switches_prec_CIs}, {HHMM_switches_prec_means+HHMM_switches_prec_CIs})')
print(f'HHMM Switches recall: {HHMM_switches_rec_means}  95% CI: ({HHMM_switches_rec_means-HHMM_switches_rec_CIs}, {HHMM_switches_rec_means+HHMM_switches_rec_CIs})')
print(f'HHMM Switches F1 Score: {HHMM_switches_F1_means}  95% CI: ({HHMM_switches_F1_means-HHMM_switches_F1_CIs}, {HHMM_switches_F1_means+HHMM_switches_F1_CIs})')
print(f'HHMM Switches MCC: {HHMM_switches_MCC_means}  95% CI: ({HHMM_switches_MCC_means-HHMM_switches_MCC_CIs}, {HHMM_switches_MCC_means+HHMM_switches_MCC_CIs})')

print('HHMM\n', total_confusion_matrix_off_task)
print(f'HHMM Off-Task Precision: {HHMM_off_task_prec_means}  95% CI: ({HHMM_off_task_prec_means-HHMM_off_task_prec_CIs}, {HHMM_off_task_prec_means+HHMM_off_task_prec_CIs})')
print(f'HHMM Off-Task recall: {HHMM_off_task_rec_means}  95% CI: ({HHMM_off_task_rec_means-HHMM_off_task_rec_CIs}, {HHMM_off_task_rec_means+HHMM_off_task_rec_CIs})')
print(f'HHMM Off-Task F1 Score: {HHMM_off_task_F1_means}  95% CI: ({HHMM_off_task_F1_means-HHMM_off_task_F1_CIs}, {HHMM_off_task_F1_means+HHMM_off_task_F1_CIs})')
print(f'HHMM Off-Task MCC: {HHMM_off_task_MCC_means}  95% CI: ({HHMM_off_task_MCC_means-HHMM_off_task_MCC_CIs}, {HHMM_off_task_MCC_means+HHMM_off_task_MCC_CIs})')

# Indexed by num_slack_frames and then by measure of performance (from interrater_reliability.py)
# Number of Frames used to measure Inter-rater reliability: 25075
# This first set of results omits frames classified as Off Task by either coder (dotted black line in plots)
interrater_switch_performance_no_off_task = [{ 'prec/rec/F1' : 0.553980678372, 'MCC' : 0.540290693059 },
                                             { 'prec/rec/F1' : 0.757650045421, 'MCC' : 0.750207642797 },
                                             { 'prec/rec/F1' : 0.833007711229, 'MCC' : 0.827876914201 }]
interrater_switch_performance_95CI_no_off_task = [{ 'prec/rec/F1' : 0.0541262216345, 'MCC' : 0.0542673928783 },
                                                  { 'prec/rec/F1' : 0.0466594230506, 'MCC' : 0.0471372108679 },
                                                  { 'prec/rec/F1' : 0.0406121784382, 'MCC' : 0.0411041823982 }]
interrater_average_no_off_task = 0.9556202255365588
interrater_average_95CI_no_off_task = 0.0224242939696

# Inter-rater results including frames classified as Off Task (dotted white line in plots)
interrater_switch_performance = [{ 'prec/rec/F1' : 0.219524334621, 'MCC' : 0.172656208336 },
                                 { 'prec/rec/F1' : 0.563307726573, 'MCC' : 0.536115618573 },
                                 { 'prec/rec/F1' : 0.713246989088, 'MCC' : 0.694636469785 }]
interrater_switch_performance_95CI = [{ 'prec/rec/F1' : 0.045071766976, 'MCC' : 0.0411545324437 },
                                      { 'prec/rec/F1' : 0.0540062688792, 'MCC' : 0.054302230802 },
                                      { 'prec/rec/F1' : 0.0492444774202, 'MCC' : 0.0501500003695 }]
interrater_average = 0.8500231803430691
interrater_average_95CI = 0.0388786360775
