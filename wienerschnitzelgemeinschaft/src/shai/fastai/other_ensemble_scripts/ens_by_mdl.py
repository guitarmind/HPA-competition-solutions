import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

SAMPLE = '../input/sample_submission.csv'

label_names = {
        0:  "Nucleoplasm",
        1:  "Nuclear membrane",
        2:  "Nucleoli",
        3:  "Nucleoli fibrillar center",
        4:  "Nuclear speckles",
        5:  "Nuclear bodies",
        6:  "Endoplasmic reticulum",
        7:  "Golgi apparatus",
        8:  "Peroxisomes",
        9:  "Endosomes",
        10:  "Lysosomes",
        11:  "Intermediate filaments",
        12:  "Actin filaments",
        13:  "Focal adhesion sites",
        14:  "Microtubules",
        15:  "Microtubule ends",
        16:  "Cytokinetic bridge",
        17:  "Mitotic spindle",
        18:  "Microtubule organizing center",
        19:  "Centrosome",
        20:  "Lipid droplets",
        21:  "Plasma membrane",
        22:  "Cell junctions",
        23:  "Mitochondria",
        24:  "Aggresome",
        25:  "Cytosol",
        26:  "Cytoplasmic bodies",
        27:  "Rods & rings"
    }

column_sum = []

def expand(csv):
    sub = pd.read_csv(csv)
    print(csv, sub.isna().sum())

    sub = sub.replace(pd.np.nan, '0')
    sub[f'target_vec'] = sub['Predicted'].map(lambda x: list(map(int, x.strip().split())))
    for i in range(28):
        sub[f'{label_names[i]}'] = sub['Predicted'].map(
                 lambda x: 1 if str(i) in x.strip().split() else 0)
    sub = sub.values
    sub = np.delete(sub, [1, 2], axis=1)

    a = sub[:, 1:]
    print(a.sum(), a.sum(axis=0))
    column_sum.append( a.sum(axis=0))
    return sub

#======================================================================================================================
# Voting
#====================================================================================================================

mdl = 'sub/wrn_long'

df_1 =  expand(mdl + '/protein_classification.csv') #497
df_2 =  expand(mdl + '/protein_classification_v.csv') # 485
df_3 =  expand(mdl + '/protein_classification_t.csv') # 491
df_4 =  expand(mdl + '/protein_classification_f.csv') # 488
df_5 =  expand(mdl + '/protein_classification_c.csv') # 483
df_6 =  expand(mdl + '/protein_classification_05.csv') # 483
#_tmp = expand('sub_ensemble/en_res34+swa+grn_6mdl_4.6.csv')

list =[0,1,2,3,4,5, 6]
for i in list:
    plt.plot(column_sum[i], label=i)
plt.legend()
plt.show()

#==========================================================================================================================

sum = df_1[:, 1:] + df_2[:, 1:] + df_3[:, 1:] + df_4[:, 1:] + df_5[:, 1:]  + df_6[:, 1:]

vote_sub = np.where(sum >= 4, 1, 0)

#======================================================================================================================
# prepare submission format
#======================================================================================================================
submit = pd.read_csv(SAMPLE)
prediction = []

for row in tqdm(range(submit.shape[0])):

    str_label = ''

    for col in range(vote_sub.shape[1]):
        if (vote_sub[row, col] < 1):
            str_label += ''
        else:
            str_label += str(col) + ' '
    prediction.append(str_label.strip())

submit['Predicted'] = np.array(prediction)
submit.to_csv('sub/sub_ensemble/en_res34+swa+grn_6mdl_4.6.csv', index=False)



print('done')