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
sub_dir = 'subs/'

df_1 =  expand(sub_dir + 'res34_sub_swa/protein_classification_f.csv') #497
df_2 =  expand(sub_dir + 'res34_sub_swa/protein_classification_v.csv') # 485
df_3 =  expand(sub_dir + 'res34_sub/protein_classification_t.csv') # 491
#df_4 =  expand(sub_dir + 'res34_sub/protein_classification_f.csv') # 488
df_4 =  expand(sub_dir + 'res50xt_sub/protein_classification_c.csv') # 483
df_5 =  expand(sub_dir + 'res101xt_sub/protein_classification_05.csv') # 492
df_6 =  expand(sub_dir + 'res101xt_sub_swa/protein_classification_05.csv') # 493
df_7 =  expand(sub_dir + 'wrn_sub_long/protein_classification_v.csv') # 498
df_8 =  expand( 'sub_dir_team/ens43.csv') # 504
df_9 =  expand( 'sub_dir_team/submission_loss_0_lb_dist_adjusted_8tta.csv') # 476

# 1:4 similar

# list =[0,1,2,3,4,5,6,7,8]
# for i in list:
#     x = np.arange(0, 28, 1)
#     plt.step(x, column_sum[i] , label=i)
# plt.legend()
# plt.grid(True)
# plt.show()

#=======================================================================================================================


sum = df_1[:, 1:] + df_2[:, 1:] + df_3[:, 1:] + df_4[:, 1:] + df_5[:, 1:]  + df_6[:, 1:] + df_7[:, 1:] +  df_8[:, 1:] + df_9[:, 1:]


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
submit.to_csv('sub_dir_team/en_v1.6team_4.9.csv', index=False)

print('done')