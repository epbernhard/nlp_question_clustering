import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import re
import string
import nltk
import numpy as np
import torch
from sklearn import model_selection, cluster, decomposition, metrics
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from matplotlib.font_manager import FontProperties
font0 = FontProperties(family = 'serif', variant = 'small-caps', size = 15)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

stop_words = set(nltk.corpus.stopwords.words('english'))

def perform_embedding(text):
    """ perform the embedding of the questions """

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True, # Whether the model returns all hidden-states.
                                     )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    sentence_embedded = []
    max_len = 0
    for text_i in text:

        marked_text = "[CLS] " + text_i + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(marked_text)
        
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Mark each of the 22 tokens as belonging to sentence "1".
        # (We assume single sentences although few questions (~5/98) are split in two sentences,
        # but it's no problem because they are the same subject).
        segments_ids = [1] * len(tokenized_text)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers. 
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        # average over the sentence to create a single vector of 1 x 768
        token_vecs = hidden_states[-2][0]
        token_vecs_avg = torch.mean(token_vecs, dim=0).tolist()

        # # Calculate the average of all 22 token vectors.

        # pdb.set_trace()


        # # Use the sum of the last 4 hidden layers for embedding
        # # Stores the token vectors, with shape [22 x 768]
        # token_vecs_sum = []

        # # For each token in the sentence...
        # for token in token_embeddings:

        #     # Sum the vectors from the last four layers.
        #     sum_vec = torch.sum(token[-4:], dim=0)
            
        #     # Use `sum_vec` to represent `token`.
        #     token_vecs_sum += sum_vec.tolist()

        sentence_embedded.append(token_vecs_avg)

    #     if len(token_vecs_sum) > max_len:
    #         max_len = len(token_vecs_sum)

    # # padding
    # sentence_embedded_padded = []
    # for sentence_embedded_i in sentence_embedded:
    #     if len(sentence_embedded_i) < max_len:
    #         n_zeros = max_len - len(sentence_embedded_i)
    #         sentence_embedded_i += [0] * n_zeros

    #     sentence_embedded_padded.append(sentence_embedded_i)

    return sentence_embedded


def pos_tagger(nltk_tag):
    """ Return the part of speech tag"""

    if nltk_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif nltk_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:         
        return None

def clean_text(text):
    """Clean the questions"""

    text_clean = ["" for i in range(len(text))]

    print('Cleaning the questions...')
    with tqdm(total=len(text)) as pbar:
        for i, text_i in enumerate(text):

            try:

                # Remove lower case and Unicode characters
                text_i = text_i.lower().encode('ascii', 'ignore').decode()

                # Remove:
                to_remove = [
                             r"@\S+", #mention
                             r"https*\S+", #link
                             r"#\S+", #hashtag
                             r"\'\w+", #ticks
                             r"[%s]", #punctuation
                             r"\w*\d+\w*", #number
                             r"\s{2,}", #over space
                             r" +" # first/last blank
                             ]

                for to_remove_i in to_remove:
                    if to_remove_i == "[%s]":
                        text_i = re.sub(to_remove_i % re.escape(string.punctuation), \
                                                   ' ', text_i)
                    elif to_remove_i == " +":
                        text_i = text_i.strip()
                    else:
                        text_i = re.sub(to_remove_i, ' ', text_i)

                # POS tagging (tagging words based on their part of speech [verbs, nouns,...]
                pos_tag = nltk.pos_tag(nltk.tokenize.word_tokenize(text_i)) # Generate list of tokens
                wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tag))

                # lemmatise based on POS tagging
                lemmatizer = nltk.stem.WordNetLemmatizer()
                lem_text = []
                for word, tag in wordnet_tagged:
                    if tag is None:
                        # if there is no available tag, append the token as is
                        lem_text.append(word)
                    else:       
                        # else use the tag to lemmatize the token
                        lem_text.append(lemmatizer.lemmatize(word, tag))
                text_i = " ".join(lem_text)

                # Remove stop words                
                text_i = ' '.join([word_i for word_i in text_i.split(' ')
                                    if word_i not in stop_words])

                # remove single letter words (i, e, a)
                text_i = [word for word in text_i.split(' ') if len(word) != 1]

                text_clean[i] = ' '.join(text_i)

            except AttributeError:

                pass

            except:

                raise ValueError("It looks like the string '{}'"+\
                                 "could not be handled.".format(text_i))

            pbar.update(1)

    return text_clean

# Get the questions
df_questions = pd.read_csv('../data/questions/questions.csv', index_col = 0)
df_questions.info()
print(df_questions.head())

# clean the questions
questions_cleaned = clean_text(df_questions['text'].values)
df_questions['questions_cleaned'] = questions_cleaned

# embedded sentences (BERT)
questions_embedded = perform_embedding(df_questions['text'].values)

## cluster the questions by subjects
n_clusters = [2, 3, 4, 5, 10,20] # number of clusters considered
parameter_grid = model_selection.ParameterGrid({'n_clusters': n_clusters})
best_score = -1.
kmeans_model = cluster.KMeans(init = 'k-means++', n_init = 50)
silhouette_scores = []

# run Kmeans on all the different values of n_clusters
# and calculate the silhouette score
for p in parameter_grid:
    kmeans_model.set_params(**p)
    kmeans_model.fit(questions_embedded)
    ss = metrics.silhouette_score(questions_embedded, kmeans_model.labels_)
    silhouette_scores.append(ss)
    if ss > best_score:
        best_score = ss
        best_grid = p

# Optimum number of clusters
optimum_n_clusters = n_clusters[np.argmax(silhouette_scores)]

# Plot the silhouette score
df_silhouette = pd.DataFrame({'n_clusters': n_clusters, 'silhouette_scores': silhouette_scores})
fig, axs = plt.subplots(figsize = (12, 6))
sns.barplot(x = 'n_clusters', y = 'silhouette_scores', data=df_silhouette, errwidth = 0.3,
            palette = sns.color_palette("flare", len(n_clusters)))
axs.set_ylabel('Silhouette Scores', fontproperties = font0)
axs.set_xlabel('Number of Clusters', fontproperties = font0)
axs.text(0.75, 0.95,'The optimum number of clusters is {}'.format(optimum_n_clusters), 
         horizontalalignment='center', verticalalignment='center',
         transform = axs.transAxes, fontproperties = font0)
fig.suptitle('Silhouette scores obtained for different numbers of Kmeans clusters', fontproperties = font0)
plt.tight_layout()
plt.show()

# fitting the KMeans
kmeans = cluster.KMeans(n_clusters=optimum_n_clusters, init = 'k-means++', n_init = 50)
kmeans.fit(questions_embedded)

# Visualise the result of the Kmean using PCA to reduce dimensions to 2.
reduced_data = decomposition.PCA(n_components=2).fit_transform(questions_embedded)
df_kmeans = pd.DataFrame({'x': np.array([x[0] for x in reduced_data]), 
                          'y': np.array([x[1] for x in reduced_data]),
                          'subjects': kmeans.labels_})

markers = ['o', 's', 'h', 'd']
fig, axs = plt.subplots(figsize = (8, 6))
sns.scatterplot(data=df_kmeans, x="x", y="y", hue="subjects", style="subjects",
                markers = markers[0:optimum_n_clusters], palette="Set2")
axs.set_xlabel('PCA axis one', fontproperties = font0)    
axs.set_ylabel('PCA axis two', fontproperties = font0)
fig.suptitle('2D projection of the clusters of subjects', fontproperties = font0)
plt.tight_layout()
plt.show()

# save the results
df_questions['subject'] = kmeans.labels_

## Get the corresponding scores
df_scores = pd.read_csv('../data/questions/scores.csv', index_col = 0)
df_scores.info()
print(df_scores.head())

# Merge the two data frames
df_questions_scores = pd.merge(df_questions, df_scores,
							   left_on = '_id',
							   right_on = 'question', 
							   suffixes=("_questions", "_scores"))
df_questions_scores.info()
print(df_questions_scores.head())

# retain what we need for now
_id = np.arange(0, len(df_questions_scores)).tolist()
df_questions_scores['_id'] = _id
df_save = df_questions_scores[['_id', 
                               'text', 
                               'questions_cleaned', 
                               'criterion', 
                               'score', 
                               'subject']].to_csv('../tmp/questions_preanal.csv', index = False)



pdb.set_trace()