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
sub_name = []

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
    unique, counts = np.unique(a, return_counts=True)
    print('Unique counts:',np.asarray((unique, counts)).T)
    print('Total labels:{} Class-wise:{}'.format(a.sum(), a.sum(axis=0)))
    column_sum.append( a.sum(axis=0))
    sub_name.append(csv)
    return sub
#======================================================================================================================
# Voting
#====================================================================================================================
sub_dir = 'sub_dir_team/'

df_1 =  expand('sub_dir_team/leak_brian_tommy_en_res34swa_re50xt_re101xtswa_wrn_4.8_562.csv')
df_2 =  expand( 'sub_dir_team/Christof_blend_4_580.csv')
df_3 =  expand('sub_dir_team/ens85bd_russ_616.csv')
df_4 =  expand('sub_dir_team/enspreds103_12mdl_512-256_wtth0.45_leak_shai_593.csv')
df_5 =  expand('sub_dir_team/hill_m92d_dmytro_617.csv')
df_6 =  expand('sub_dir_team/voted_5_d_kevin_602.csv')
df_7 =  expand('sub_dir_team/hill_b92d_l2_615.csv')
df_8 =  expand('sub_dir_team/sub42_ensemble_598.csv')

#=======================================================================================================================
list =[0,1,2,3,4,5,6, 7]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
w=0
for i in list:
    x = np.arange(0, 28, 1)
    plt.bar(x+w, column_sum[i],width = 0.1, color = colors[i], label=sub_name[i], )
    w=w+0.11
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

#=======================================================================================================================
sim = df_1*1 + df_2*1 + df_3*1 + df_4*1 + df_5*1 + df_6*1 + df_7*1 + df_8*1
sim = sim[:, 1:]
unique, counts = np.unique(sim, return_counts=True)
#=======================================================================================================================
sum = df_1[:, 1:]*1 + \
      df_2[:, 1:]*3 + \
      df_3[:, 1:]*3 + \
      df_4[:, 1:]*2 + \
      df_5[:, 1:]*3 + \
      df_6[:, 1:]*2 + \
      df_7[:, 1:]*1 + \
      df_8[:, 1:]*1

vote = 8

vote_sub0  = np.where(sum[:,0]  >= vote,    1, 0)      #high
vote_sub1  = np.where(sum[:,1]  >= vote,    1, 0)
vote_sub2  = np.where(sum[:,2]  >= vote,    1, 0)
vote_sub3  = np.where(sum[:,3]  >= vote,    1, 0)
vote_sub4  = np.where(sum[:,4]  >= vote,    1, 0)
vote_sub5  = np.where(sum[:,5]  >= vote,    1, 0)
vote_sub6  = np.where(sum[:,6]  >= vote,    1, 0)
vote_sub7  = np.where(sum[:,7]  >= vote,    1, 0)
vote_sub8  = np.where(sum[:,8]  >= vote,    1, 0)      #low
vote_sub9  = np.where(sum[:,9]  >= vote-2,  1, 0)      #low
vote_sub10 = np.where(sum[:,10] >= vote-2,  1, 0)      #low
vote_sub11 = np.where(sum[:,11] >= vote,    1, 0)
vote_sub12 = np.where(sum[:,12] >= vote,    1, 0)
vote_sub13 = np.where(sum[:,13] >= vote,    1, 0)
vote_sub14 = np.where(sum[:,14] >= vote,    1, 0)
vote_sub15 = np.where(sum[:,15] >= vote+4,  1, 0)      #low
vote_sub16 = np.where(sum[:,16] >= vote,    1, 0)
vote_sub17 = np.where(sum[:,17] >= vote,    1, 0)
vote_sub18 = np.where(sum[:,18] >= vote,    1, 0)
vote_sub19 = np.where(sum[:,19] >= vote,    1, 0)
vote_sub20 = np.where(sum[:,20] >= vote,    1, 0)
vote_sub21 = np.where(sum[:,21] >= vote,    1, 0)
vote_sub22 = np.where(sum[:,22] >= vote,    1, 0)
vote_sub23 = np.where(sum[:,23] >= vote,    1, 0)
vote_sub24 = np.where(sum[:,24] >= vote,    1, 0)
vote_sub25 = np.where(sum[:,25] >= vote,    1, 0)      #high
vote_sub26 = np.where(sum[:,26] >= vote,    1, 0)
vote_sub27 = np.where(sum[:,27] >= vote-3,  1, 0)      #low

vote_sub = np.column_stack((vote_sub0,vote_sub1, vote_sub2, vote_sub3,
                            vote_sub4, vote_sub5,vote_sub6, vote_sub7,
                            vote_sub8, vote_sub9,vote_sub10,vote_sub11,
                            vote_sub12, vote_sub13, vote_sub14, vote_sub15,
                            vote_sub16, vote_sub17, vote_sub18, vote_sub19,
                            vote_sub20,vote_sub21, vote_sub22, vote_sub23,
                            vote_sub24,vote_sub25,vote_sub26, vote_sub27)
                           )
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
#submit.to_csv('sub_dir_team/test.csv', index=False)
submit.to_csv('sub_dir_team/enstw38_1shai562_3russ616_3dieter580_2shai593_3dmytro617_2kevin602_1l2.615_1dmytro598_8.16_clswt.csv', index=False)

#=======================================================================================================================