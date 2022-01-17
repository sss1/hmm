"""
This script reproduces the main developmental findings, namely changes in the
proportions of time spent in each mode over Age.
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

try:
  print('Trying to load cached HHMM...')
  melted_df = pd.read_csv('cached_dev_melted_df.csv')
  joint_df = pd.read_csv('cached_dev_joint_df.csv')

except FileNotFoundError:
  print('Cached HHMM not found. Training from scratch...')

  import load_subjects as ls
  import util
  
  # Load trial data to reconstruct trials
  subjects = ls.load_all_data()
  
  # Load model parameters to reconstruct model
  model_args_df = pd.read_csv('training_results_by_age_hhmm.csv')
  # object_switch_prob wasn't exported, so calculate this
  model_args_df['Pi_DD'] = 1 - (model_args_df['Pi_DO'] + model_args_df['Pi_DI'])
  model_args_df['object_switch_prob'] = model_args_df['Pi_DD_switch']/model_args_df['Pi_DD']
  
  modes_df_as_dict = {'subject_ID': [], 'condition': [], 'D': [], 'O': [], 'I': []}
  
  for subject_ID, subject in subjects.items():
    for condition in ['shrinky', 'noshrinky']:
  
      true_means, trial_lens, observations = hhmm.format_data(subjects[subject_ID].experiments[condition])
      valid_data_mask = hhmm.get_valid_data_mask(trial_lens, observations).numpy()
  
      model_args = util.format_model_args_from_row(subject_ID, model_args_df, condition)
      HHMM_MLE = hhmm.get_MLE_states(true_means, trial_lens, observations, model_args).numpy()
  
      HHMM_MLE = HHMM_MLE.flatten()[valid_data_mask.flatten()]
  
      modes_df_as_dict['subject_ID'].append(int(subject_ID))
      modes_df_as_dict['condition'].append(condition)
      modes_df_as_dict['D'].append(np.mean(HHMM_MLE < 7))
      modes_df_as_dict['O'].append(np.mean(HHMM_MLE == 7))
      modes_df_as_dict['I'].append(np.mean(HHMM_MLE == 8))
  
  modes_df = pd.DataFrame(modes_df_as_dict).set_index(['subject_ID', 'condition'])
  joint_df = (model_args_df.set_index(['subject_ID', 'condition'])
                           .join(modes_df, how='inner')
                           .reset_index())[['subject_ID', 'condition', 'age', 'D', 'O', 'I']]
  melted_df = (joint_df.melt(id_vars=['subject_ID', 'condition', 'age'],
                             value_vars=['D', 'O', 'I'],
                             var_name='Mode',
                             value_name='Proportion of Time')
                       .rename(columns={'age': 'Age (years)',
                                        'condition': 'Condition'})
                       .replace({'D': 'Distractible',
                                 'O': 'Optimally Engaged',
                                 'I': 'Inattentive',
                                 'shrinky': 'Exogenous',
                                 'noshrinky': 'Endogenous'}))

  print('Caching trained HHMM...')  
  melted_df.to_csv('cached_dev_melted_df.csv')
  joint_df.to_csv('cached_dev_joint_df.csv')

melted_df = pd.read_csv('cached_dev_melted_df.csv')
joint_df = pd.read_csv('cached_dev_joint_df.csv')

melted_df = melted_df[melted_df['Condition'] == 'Endogenous']
melted_df = melted_df.replace({'Optimally Engaged': 'Task Engaged'})

plt.rcParams.update({'font.size': 20})
sns.lmplot(x='Age (years)', y='Proportion of Time', col='Mode',
           data=melted_df, legend=False)
plt.ylim((-0.05, 1.05))
# plt.legend(loc='upper right')
plt.tight_layout()

print(smf.ols(formula='D ~ age*condition', data=joint_df).fit().summary())
print(smf.ols(formula='O ~ age*condition', data=joint_df).fit().summary())
print(smf.ols(formula='I ~ age*condition', data=joint_df).fit().summary())
plt.show()
