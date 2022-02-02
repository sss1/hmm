import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.formula.api as smf

import hhmm
import load_subjects as ls
import util

cache_filename = 'trackit_performance_results.csv'

try:
  print('Trying to load pre-calculated results...')
  df = pd.read_csv(cache_filename)

except FileNotFoundError:
  print('Pre-calculated results not found, calulating from scratch...')

  trials_to_include = range(1, 11)
  
  # Load trial data to reconstruct trials
  subjects = ls.load_all_data()
  
  # Load model parameters to reconstruct model
  model_args_df = pd.read_csv('training_results_by_age_hhmm.csv')
  # object_switch_prob wasn't exported, so calculate this
  model_args_df['Pi_DD'] = 1 - (model_args_df['Pi_DO'] + model_args_df['Pi_DI'])
  model_args_df['object_switch_prob'] = model_args_df['Pi_DD_switch']/model_args_df['Pi_DD']
  
  trial_df_as_dict = {'subject_ID': [], 'condition': [], 'age': [], 'D': [], 'O': [],  'I':[], 'loc_acc': [], 'mem_check': []}
  
  for subject_ID, subject in subjects.items():
    for condition in ['shrinky', 'noshrinky']:
  
      true_means, trial_lens, observations = hhmm.format_data(subjects[subject_ID].experiments[condition])
      valid_data_mask = hhmm.get_valid_data_mask(trial_lens, observations).numpy()
      experiment_metadata = subject.experiments['shrinky'].datatypes['trackit'].metadata
  
      model_args = util.format_model_args_from_row(subject_ID, model_args_df, condition)
      HHMM_MLE = hhmm.get_MLE_states(true_means, trial_lens, observations, model_args).numpy()
  
      for trial in trials_to_include:
        HHMM_MLE_trial = HHMM_MLE[trial-1, :][valid_data_mask[trial-1, :]]
        trial_metadata = subject.experiments[condition].datatypes['trackit'].trials[trial].trial_metadata
  
        trial_df_as_dict['subject_ID'].append(int(subject_ID))
        trial_df_as_dict['condition'].append(condition)
        trial_df_as_dict['age'].append(float(experiment_metadata['Age']))
        trial_df_as_dict['D'].append(np.mean(HHMM_MLE_trial < 7))
        trial_df_as_dict['O'].append(np.mean(HHMM_MLE_trial == 7))
        trial_df_as_dict['I'].append(np.mean(HHMM_MLE_trial == 8))
        trial_df_as_dict['loc_acc'].append(
            trial_metadata['gridClickCorrect'] == 'true')
        trial_df_as_dict['mem_check'].append(
            trial_metadata['lineupClickCorrect'] == 'true')
  
  df = pd.DataFrame(trial_df_as_dict)
  
  df.to_csv(cache_filename)

df = df.groupby('subject_ID')[['D', 'O', 'I', 'loc_acc']].mean()
for mode in ['D', 'O', 'I']:
  r, p = scipy.stats.pearsonr(df[mode], df['loc_acc'])
  delta = 1.96 / math.sqrt(len(df) - 3)
  lower = math.tanh(math.atanh(r) - delta)
  upper = math.tanh(math.atanh(r) + delta)
  print(f'Pearson correlation between {mode} and loc_acc: {r:.2f}  ({lower:.2f}, {upper:.2f})    p-value: {p:.3f}')
  sns.lmplot(x=mode, y='loc_acc', data=df, legend=False)
  plt.xlim((0, 1))
  plt.ylim((0, 1))

plt.show()
