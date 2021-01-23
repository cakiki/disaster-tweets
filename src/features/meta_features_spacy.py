"""
Set of Sklearn transformer implementations that expect spacy documents as inputs and output vector spaces.
See the meta_feature_exploration.ipynb notebook for an in-depth evaluation and usage examples on each of the
transformers.
"""

from sklearn import base, preprocessing, feature_extraction, feature_selection
from sklearn import svm
from sklearn import ensemble
import numpy as np
import re
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn

# Transformer class for standard TFIDF matrix extraction
class TfidfVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to create TF-IDF vector spaces from spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents.  Outputs a feature matrix of shape (n_docs, n_words).

    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    :param remove_stop: Whether to remove stop words or not (using the spacy is_stop flag on the tokens)
    :type remove_stop: bool, optional
    :param remove_non_alpha: Whether to remove tokens that do not consist of alphabetic characters (using the spacy is_alpha flag)
    :type remove_non_alpha: bool, optional
    """

    def __init__(self, ngram_range=(1, 1), remove_stop=False, remove_non_alpha=False):
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
        self.remove_stop = remove_stop
        self.remove_non_alpha = remove_non_alpha
    
    def fit(self, X_sp, y=None):
        X_trans = np.array([" ".join([token.text for token in x if self.include_token(token)]) for x in X_sp])
        self.vectorizer.fit(X_trans, y)
        return self
    
    def transform(self, X_sp):
        X_trans = np.array([" ".join([token.text for token in x if self.include_token(token)]) for x in X_sp])
        return self.vectorizer.transform(X_trans)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    def include_token(self, token):
        if self.remove_stop and token.is_stop:
            return False
        if self.remove_non_alpha and not token.is_alpha:
            return False
        return True
         

# Transformer class that counts the number of links in a spacy sentence
class LinkCounter(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to count the number of links in a spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, 1)`.
    """

    def __init__(self):
        pass
    
    def fit(self, X_sp, y):
        return self
    
    def transform(self, X_sp):
        return np.array([self.count_links(x.text) for x in X_sp]).reshape((len(X_sp), 1))

    def count_links(self, txt):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)
        return len(urls)
    
    def get_feature_names(self):
        return ['link_count']

# Transformer class that outputs the text length of spacy sentences 
class TextLength(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to compute the length of spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, 1)`. The length is either interpreted as the number of characters level or the number of tokens.

    :param mode: How to interpret the length of the tweet. Can be either `characters` or `tokens`
    :type mode: string, optional
    """

    def __init__(self, mode='characters'):
        self.mode = mode
    
    def fit(self, X_sp, y):
        return self
    
    def transform(self, X_sp):
        if self.mode == 'tokens':
            return np.array([len(x) for x in X_sp]).reshape((len(X_sp), 1))
        else:
            return np.array([len(x.text) for x in X_sp]).reshape((len(X_sp), 1))

    def get_feature_names(self):
        return ['text_length']

# Transformer class that outputs the number of named entities per spacy sentence 
class NumNamedEntities(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to count the number of named entities in a spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, 1)`.
    """

    def __init__(self):
        pass
    
    def fit(self, X_sp, y):
        return self
    
    def transform(self, X_sp):
        return np.array([len(x.ents) for x in X_sp]).reshape((len(X_sp), 1))

    def get_feature_names(self):
        return  ['num_named_entities']

# Transformer class that outputs a term frequency matrix of Entity labels.
class EntityTagCountVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to create TF-IDF vector spaces from entity types in spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, n_types)` (typs are ORG, GPE, MONEY, ...). See https://spacy.io/api/annotation#named-entities for possible types.

    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    """

    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    
    def fit(self, X_sp, y=None):
        X_trans = np.array([" ".join([ent.label_ for ent in x.ents]) for x in X_sp])
        self.vectorizer.fit(X_trans, y)
        return self
    
    def transform(self, X_sp):
        X_trans = np.array([" ".join([ent.label_ for ent in x.ents]) for x in X_sp])
        return self.vectorizer.transform(X_trans)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

# Transformer class that outputs a term frequency matrix of POS tags.
# You can specify either pos_type='universal' or pos_type='penn_treebank'
class PosTagCountVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to create TF-IDF vector spaces from Part-of-Speech tags in spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, n_pos)` (pos are PROPN, VERB, NOUN, ...). See https://spacy.io/api/annotation#pos-tagging for possible pos values.

    :param pos_type: The tag set to use. May be either `universal` or `penn_treebank` for an english-specific tag set. Defaults to `universal`
    :type pos_type: string, optional
    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    """

    def __init__(self, pos_type='universal', ngram_range=(1, 1)):
        self.pos_type = pos_type
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    
    def fit(self, X_sp, y=None):
        X_trans = np.array([" ".join([token.tag_ if self.pos_type == 'penn_treebank' else token.pos_ for token in x]) for x in X_sp])
        self.vectorizer.fit(X_trans, y)
        return self
    
    def transform(self, X_sp):
        X_trans = np.array([" ".join([token.tag_ if self.pos_type == 'penn_treebank' else token.pos_ for token in x]) for x in X_sp])
        return self.vectorizer.transform(X_trans)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

# Transformer class that outputs a term frequency matrix of POS tags.
class DependencyTagCountVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to create TF-IDF vector spaces from syntactic dependency tags in spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, n_dep)` (dep are nsubj, dobj, pobj, ...). See https://spacy.io/api/annotation#dependency-parsing for possible dependencies values.

    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    """

    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    
    def fit(self, X_sp, y=None):
        X_trans = np.array([" ".join([token.dep_ for token in x]) for x in X_sp])
        self.vectorizer.fit(X_trans, y)
        return self
    
    def transform(self, X_sp):
        X_trans = np.array([" ".join([token.dep_ for token in x]) for x in X_sp])
        return self.vectorizer.transform(X_trans)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

# Transformer class that outputs a term frequency matrix of Emphasis tags:
# CAPITALIZED, HASHTAG, TARGET, EXCLAMATION_MARK, QUESTION_MARK or NOT_EMPHASIZED
class EmphasisCountVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to create TF-IDF vector spaces from emphasis tags in spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, n_emph)`. The folloing tags may occur:
    * `CAPITALIZED`: a token is capizalized.
    * `HASHTAG`: a token is prefixed by #.
    * `TARGET`: a token is user mention prefixed by @.
    * `ELONGATED`: a token is an elongated word (e.g. huuuuuge).
    * `NOT_EMPHASIZED`: a token is not emphasized.
    
    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    """

    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    
    def fit(self, X_sp, y=None):
        X_trans = np.array([" ".join([self.determine_tag(token) for token in x]) for x in X_sp])
        self.vectorizer.fit(X_trans, y)
        return self
    
    def transform(self, X_sp):
        X_trans = np.array([" ".join([self.determine_tag(token) for token in x]) for x in X_sp])
        return self.vectorizer.transform(X_trans)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

    def determine_tag(self, token):
        if token.shape_.isupper():
            return "CAPITALIZED"
        if "#" in token.text:
            return "HASHTAG"
        if "@" in token.text:
            return "TARGET"
        if token.pos_ != "NUM" and re.search(r'(\w)\1+\1+', token.text) is not None:
            return "ELONGATED"
        return "NOT_EMPHASIZED"

# Transformer class that outputs a term frequency matrix of punctuation tags:
# EXCLAMATION_MARK, QUESTION_MARK, COLON_MARK, QUOTATION_MARK, ELLIPSIS_MARK, ASTERISK_MARK or NO_PUNCTUATATION
class PunctuationCountVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to create TF-IDF vector spaces from punctuation tags in spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, n_tags)`. The following punctuation tags may occur:
    * `EXCLAMATION_MARK`
    * `QUESTION_MARK`
    * `COLON_MARK`
    * `QUOTATION_MARK`
    * `ELLIPSIS_MARK`
    * `ASTERISK_MARK`
    * `NO_PUNCTUATATION`
    
    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    """
    
    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    
    def fit(self, X_sp, y=None):
        X_trans = np.array([" ".join([self.determine_punct(token) for token in x]) for x in X_sp])
        self.vectorizer.fit(X_trans, y)
        return self
    
    def transform(self, X_sp):
        X_trans = np.array([" ".join([self.determine_punct(token) for token in x]) for x in X_sp])
        return self.vectorizer.transform(X_trans)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

    def determine_punct(self, token):
        if  "!" in token.text:
            return "EXCLAMATION_MARK"
        if  "?" in token.text:
            return "QUESTION_MARK"
        if  ":" in token.text:
            return "COLON_MARK"
        if  '"' in token.text:
            return "QUOTATION_MARK"
        if  "'" in token.text:
            return "QUOTATION_MARK"
        if  "..." in token.text:
            return "ELLIPSIS_MARK"
        if  "*" in token.text:
            return "ASTERISK_MARK"
        return "NO_PUNCTUATATION"

# Transformer class that outputs a term frequency matrix of common emoticons.
class EmojiCountVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to create TF-IDF vector spaces from emoji tags in spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, n_tags)`. The possible tags can be found in `/data/external/wikipedia/emoticons.json`. If none of the smilies for a token matches an entry therein, the token is tagged with `NO_EMOTICON`.
    
    :param emoticon_dict: The JSON dictionary that contains the possible emoticons with their corresponding tags
    :type emoticon_dict: string, optional
    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    """

    def __init__(self, emoticon_dict='../data/external/wikipedia/emoticons.json', ngram_range=(1, 1)):
        # Load emoticon list
        with open(emoticon_dict) as json_file:
            self.emoticons = json.load(json_file)
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    
    def fit(self, X_sp, y=None):
        X_trans = np.array([" ".join([self.find_emo_tag(token) for token in x]) for x in X_sp])
        self.vectorizer.fit(X_trans, y)
        return self
    
    def transform(self, X_sp):
        X_trans = np.array([" ".join([self.find_emo_tag(token) for token in x]) for x in X_sp])
        return self.vectorizer.transform(X_trans)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

    def find_emo_tag(self, token):
        for entry in self.emoticons:
            if token.text in entry['icons']:
                return entry['meaning']
        return 'NO_EMOTICON'

# Transformer class that outputs a term frequency matrix of common acronyms.
class AcronymCountVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to create TF-IDF vector spaces from common social media slang tags in spacy documents.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, n_tags)`. The possible tags can be found in `/data/external/slang/acronyms.json`. If none of the acronyms for a token matches an entry therein, the token is tagged with `NO_ACRONYM`.
    
    :param slang_dict: The JSON dictionary that contains the possible slang acronyms with their corresponding tags
    :type slang_dict: string, optional
    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    """

    def __init__(self, slang_dict='../data/external/slang/acronyms.json', ngram_range=(1, 1)):
        # Load acronym list
        with open(slang_dict) as json_file:
            self.acronyms = json.load(json_file)
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
    
    def fit(self, X_sp, y=None):
        X_trans = np.array([" ".join([self.find_acro_tag(token) for token in x]) for x in X_sp])
        self.vectorizer.fit(X_trans, y)
        return self
    
    def transform(self, X_sp):
        X_trans = np.array([" ".join([self.find_acro_tag(token) for token in x]) for x in X_sp])
        return self.vectorizer.transform(X_trans)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

    def find_acro_tag(self, token):
        if token.text.upper() in self.acronyms:
            return token.text.upper()
        return 'NO_ACRONYM'

# Transformer class that outputs sentiment scores using the TextBlob analyzer.
# Returns feature columns for polarity in [-1, 1] and subjectivity in [-1, 1].
class TextBlobSentimentVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to compute sentiment scores using the TextBlob Sentiment analyzer.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, 2)`. The 2 feature columns are (in order) a polarity score in [-1, 1] and a subjectivity in [-1, 1]. Details on the analyzer can be found at https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X_sp, y=None):
        return self
    
    def transform(self, X_sp):
        X_sents = np.array([TextBlob(" ".join([token.text for token in x])).sentiment for x in X_sp])
        X_polarity, X_subjectivity = X_sents[:, 0], X_sents[:, 1]
        return np.hstack((X_polarity[:, np.newaxis], X_subjectivity[:, np.newaxis]))

    def get_feature_names(self):
        return ['textblob_polarity', 'textblob_subjectivity']

# Transformer class that outputs sentiment scores using the Vader analyzer.
# Returns feature columns for compund sentiment in [-1, 1], positive sentiment probability in [0, 1],
# negative sentiment probability in [0, 1] and neutral sentiment probability in [0, 1].
class VaderSentimentVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to compute sentiment scores using the Vader Sentiment analyzer.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, 4)`. The 4 feature columns are (in order) a compund score in [-1, 1], the probability for positive sentiment in [0, 1], the probability for negative sentiment in [0, 1] and the probability for neural sentiment in [0, 1]. Details on the analyzer can be found at https://pypi.org/project/vaderSentiment.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def fit(self, X_sp, y=None):
        return self
    
    def transform(self, X_sp):
        sents = [self.analyzer.polarity_scores(" ".join([token.text for token in x])) for x in X_sp]
        X_comp = np.array([sen["compound"] for sen in sents])
        X_pos = np.array([sen["pos"] for sen in sents])
        X_neg = np.array([sen["neg"] for sen in sents])
        X_neu = np.array([sen["neu"] for sen in sents])
        return np.hstack((X_comp[:, np.newaxis], X_pos[:, np.newaxis], X_neg[:, np.newaxis], X_neu[:, np.newaxis]))

    def get_feature_names(self):
        return ['vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']

# Transformer class that outputs sentiment scores using the Afinn analyzer.
class AfinnSentimentVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to compute sentiment scores using the Afinn Sentiment analyzer.
   
    Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, 1)`. The feature column is a compund score in [-N, N] where a negative integer indicates a negative sentiment and a positive integer correspondingly a positive sentiment. Details on the analyzer can be found at https://pypi.org/project/afinn.
    """

    def __init__(self):
        self.analyzer = Afinn(emoticons=True)
    
    def fit(self, X_sp, y=None):
        return self
    
    def transform(self, X_sp):
        sents = np.array([self.analyzer.score(" ".join([token.text for token in x])) for x in X_sp])
        return sents[:, np.newaxis]

    def get_feature_names(self):
        return ['afinn_score']

# Transformer class that outputs a bag of k words with highest feature selection scores.
# The score is computed according to three statistical tests (ANOVA, Mutual Information, Chi2)
# and two importance weight scores from classifiers (Linear SVM, Gradient Boosting).
# The bag of words are taken from cleaned (stop word removal, alpha word filter) and lemmatized tokens.
class KBestWordsCountVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Sklearn transformer to to create TF-IDF vector spaces from the most differentiating words in spacy documents.
   
    This transformer can be seen as a form of dimensionality reduction for a BoW approach. Expected inputs to to fit is an iterable over spacy documents. Outputs a feature matrix of shape `(n_docs, k)` where k is a parameter to select the k most differentiating words. Preprocessing consists of removal of stop words and non-alpha words (according to spacy methods is_stop and is_alpha). This transformer must be trained. For each spacy document in the input data, the following scores are computed:
    * A score from the statistical ANOVA test (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
    * A score from a mutual information test (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif)
    * A score from the Chi-Square Test (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)
    * The feautre importance scores obtained from fitting a linear support vector machine
    * The feautre importance scores obtained from fitting a gradient boosting classifier
    
    The scores for each word in the TF-IDF vector dimension are normalized and aggregated to a single score. The k dimensions of the full TF-IDF vector space with the highest aggregated scores are used to span and output a corresponding subspace.
    
    :param k: The number of words with the highest scores to select
    :type k: int
    :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted
    :type ngram_range: tuple, optional
    """

    def __init__(self, k, ngram_range=(1, 1)):
        self.k = k
        self.anova_scores = None
        self.mi_scores = None
        self.chi2_scores = None
        self.svm_scores = None
        self.gb_scores = None
        self.aggregated_scores = None
        self.k_highest_idx = None
        self.vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range)
        
    def fit(self, X_sp, y=None):
        # Clean up and lemmatize tokens
        X_trans = np.array([' '.join(token.lemma_ for token in x if not self.is_noise(token)) for x in X_sp])
        # Vectorize full vocabulary
        X_vec = self.vectorizer.fit_transform(X_trans)
        # Compute test and classifier scores
        self.anova_scores = feature_selection.SelectKBest(feature_selection.f_classif, k=20).fit(X_vec, y).scores_
        self.mi_scores = feature_selection.SelectKBest(feature_selection.mutual_info_classif, k=20).fit(X_vec, y).scores_
        self.chi2_scores = feature_selection.SelectKBest(feature_selection.chi2, k=20).fit(X_vec, y).scores_
        clf = svm.LinearSVC(C=1e-2, class_weight='balanced', random_state=42)
        self.svm_scores = np.abs(clf.fit(X_vec, y).coef_.flatten())
        clf = ensemble.GradientBoostingClassifier(n_estimators=1000, max_depth=14, learning_rate=0.5, validation_fraction=0.1, n_iter_no_change=3, random_state=42)
        self.gb_scores = np.abs(clf.fit(X_vec, y).feature_importances_.flatten())
        # Compute aggregated scores
        self.aggregated_scores = self.anova_scores/np.sum(self.anova_scores) + self.mi_scores/np.sum(self.mi_scores) + self.chi2_scores/np.sum(self.chi2_scores) + self.svm_scores/np.sum(self.svm_scores) + self.gb_scores/np.sum(self.gb_scores)
        # Sort for k highest scores
        self.k_highest_idx = np.argsort(self.aggregated_scores)[::-1][0:self.k]
        return self
    
    def transform(self, X_sp):
        # Clean up and lemmatize tokens
        X_trans = np.array([' '.join(token.lemma_ for token in x if not self.is_noise(token)) for x in X_sp])  
        # Vectorize full vocabulary
        X_vec = self.vectorizer.transform(X_trans)
        # Pick only k-highest scoring word columns
        return X_vec[:, self.k_highest_idx]

    def get_feature_names(self):
        return np.array(self.vectorizer.get_feature_names())[self.k_highest_idx]
    
    def is_noise(self, token):
        return not token.is_alpha or token.is_stop