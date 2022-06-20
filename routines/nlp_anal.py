import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from matplotlib.font_manager import FontProperties
font0 = FontProperties(family = 'serif', variant = 'small-caps', size = 15)

import pdb

#data
df_questions = pd.read_csv('../tmp/questions_preanal.csv')

# plot the average score of each of the subjects
df_avg = df_questions.groupby(['text', 'criterion_name']).mean().reset_index()
df_avg.info()

# Plot the average score in each of the subjects
fig, axs = plt.subplots(figsize = (8, 6))
sns.barplot(x = 'score', y = 'criterion_name', hue = 'subject', data = df_avg,
            orient = 'h', saturation = 0.5, errwidth = 0.3)
axs.set_ylabel('Criteria', fontproperties = font0)
axs.set_xlabel('Average Score', fontproperties = font0)
axs.tick_params(axis='both', labelcolor='k', labelsize = 10,
                width = 1, size = 20, which = 'major', direction = 'inout')
axs.tick_params(axis='both', width = 1, size = 10, which = 'minor',
                direction = 'inout')
axs.set_xlim([0., 1.])
fig.suptitle('Average score of each of the subjects and per categories', fontproperties = font0)
plt.tight_layout()
plt.savefig('../tmp/avg_scores.pdf')

# What are the subjects
    
# plot the top 10 most common words in each of the subjects
n_subjects = len(df_questions.groupby(by = 'subject'))
n_words = 10 #Numbers of top words to look at
df_reduced = df_questions.drop_duplicates(subset = ['questions_cleaned'])
most_common_words = df_questions.groupby(by = ['kmeans_labels'])['text_cleaned'].\
                    apply(lambda x: Counter(" ".join(x).split()).most_common(n_words)).\
                    reset_index().\
                    rename(columns = {'text_cleaned' : 'most_common_words'})

fig, axs = plt.subplots(1, n_subjects, figsize = (10, 6))
for i in range(0, n_subjects):
    m_c_w = [m_c_w_i[0] for m_c_w_i in most_common_words['most_common_words'][i]]
    nm_c_w = [m_c_w_i[1] for m_c_w_i in most_common_words['most_common_words'][i]]
    df_n_m_w = pd.DataFrame({'word': m_c_w, 'n_word': nm_c_w})
    sns.barplot(x = 'n_word', y = 'word', data=df_n_m_w, errwidth = 0.3,
                palette = sns.color_palette("flare", n_words), ax = axs[i],
                orient = 'h')
    axs[i].set_title('Subject {}'.format(i), fontproperties = font0)
    axs[i].set_xlabel('Counts', fontproperties = font0)
    axs[i].set_ylabel('Words', fontproperties = font0)

fig.suptitle('{} most common words in each of the subjects'.format(n_words), fontproperties = font0)
plt.tight_layout()
plt.show()


pdb.set_trace()