import pandas as pd

import hhmm
import load_subjects as ls

# LOAD DATA
subjects = ls.load_all_data()

table_as_dict = {'subject_ID': [], 'age': [], 'loss': [], 'pi_D': [], 'pi_O': [], 'pi_I': [],
                 'Pi_DD_stay': [], 'Pi_DD_switch': [], 'Pi_DO': [], 'Pi_DI': [], 'Pi_OD': [],
                 'Pi_OO': [], 'Pi_OI': [], 'Pi_ID': [], 'Pi_IO': [],
                 'Pi_II': [], 'Sigma_x': [], 'Sigma_y': []}
for subject in subjects.values():
  table_as_dict['subject_ID'].append(subject.ID)
  table_as_dict['age'].append(float(subject.experiments['shrinky'].datatypes['trackit'].metadata['Age']))
  print(f'Training model on subject {subject.ID}...')
  loss, pi, Pi, Sigma, _ = hhmm.train_model(subject.experiments['shrinky'])

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
output_csv = 'training_results_by_age_hhmm_noOE_shrinky.csv'
df.to_csv(output_csv, index=False)
