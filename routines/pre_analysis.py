import pandas as pd
import pdb
import re
import string
from tqdm import tqdm
import nltk
import numpy as np

stop_words = set(nltk.corpus.stopwords.words('english'))


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

# Get the corresponding scores
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
df_save = df_questions_scores[['_id', 'text', 'questions_cleaned', 'criterion', 'score']].to_csv('../tmp/questions_preanal.csv', index = False)



pdb.set_trace()