"""
This script prints out basic statistics of the distribution of each HHMM
parameter (across sessions).
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

import hhmm
import load_subjects as ls
import util

num_objects = 7

# Load model parameters to reconstruct model
model_args_df = pd.read_csv('training_results_by_age_hhmm.csv')
# model_args_df = model_args_df[model_args_df['condition'] == 'noshrinky']
# Some parameters weren't exported, so calculate these
model_args_df['Pi_DD'] = 1 - (model_args_df['Pi_DO'] + model_args_df['Pi_DI'])
model_args_df['object_switch_prob'] = model_args_df['Pi_DD_switch']/model_args_df['Pi_DD']
model_args_df['object_stay_prob'] = 1 - model_args_df['object_switch_prob']
model_args_df['pi_D'] = num_objects*model_args_df['pi_D']

mode_initial_distribution_args = model_args_df[['pi_D', 'pi_O', 'pi_I']]
print(mode_initial_distribution_args[model_args_df['condition'] == 'shrinky'].describe())
print(mode_initial_distribution_args[model_args_df['condition'] == 'noshrinky'].describe())

mode_transition_distribution_args = model_args_df[['Pi_DD', 'Pi_DO', 'Pi_DI', 'Pi_OD', 'Pi_OO', 'Pi_OI', 'Pi_ID', 'Pi_IO', 'Pi_II']]
print(mode_transition_distribution_args[model_args_df['condition'] == 'shrinky'].describe())
print(mode_transition_distribution_args[model_args_df['condition'] == 'noshrinky'].describe())

distractible_state_transition_distribution_args = model_args_df[['object_switch_prob', 'object_stay_prob']]
print(distractible_state_transition_distribution_args[model_args_df['condition'] == 'shrinky'].describe())
print(distractible_state_transition_distribution_args[model_args_df['condition'] == 'noshrinky'].describe())

Sigma_args = model_args_df[['Sigma_x', 'Sigma_y']]
print(Sigma_args[model_args_df['condition'] == 'shrinky'].describe())
print(Sigma_args[model_args_df['condition'] == 'noshrinky'].describe())
