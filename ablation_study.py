import hhmm
import load_subjects as ls

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# LOAD DATA
subjects = ls.load_all_data()
trials_to_include = range(1, 11)  # Omit practice trials

table_as_dict = {'subject_ID': [], 'age': [], 'condition': [], 'test_trial': [],
                 'training_log_likelihood': [], 'test_log_likelihood': [],
                 'trial_len': []}
conditions = ['shrinky', 'noshrinky']

# To fix any model parameters to 0, add their logits to this list; e.g., to
# exclude the Inattentive mode:
# to_remove = ['logit_pi_I', 'logit_Pi_OI', 'logit_Pi_DI']
to_remove = []

for subject in subjects.values():
  for trial in trials_to_include:
    training_trials = sorted(list(set(trials_to_include) - {trial}))
    test_trials = [trial]
    for condition in conditions:
      experiment = subject.experiments[condition]
      print(f'Running subject {subject.ID}, condition {condition}...')
      training_loss, _, _, _, model_args = hhmm.train_model(
          subject.experiments[condition],
          trials_to_include=training_trials,
          to_remove=to_remove)
      test_log_likelihood = hhmm.test_model(experiment,
                                            model_args,
                                            test_trials)

      table_as_dict['subject_ID'].append(subject.ID)
      table_as_dict['age'].append(
          float(experiment.datatypes['trackit'].metadata['Age']))
      table_as_dict['condition'].append(condition)
      table_as_dict['test_trial'].append(trial)
      table_as_dict['training_log_likelihood'].append(-training_loss)
      table_as_dict['test_log_likelihood'].append(test_log_likelihood)
      table_as_dict['trial_len'].append(
          experiment.datatypes['eyetrack'].trials[trial].data.shape[0])

df = pd.DataFrame(table_as_dict)
df['normalized_test_log_likelihood'] = df['test_log_likelihood'] / df['trial_len']
output_csv = 'likelihood_cross_validation_results_full.csv'
df.to_csv(output_csv, index=False)
