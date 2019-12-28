from collections import defaultdict, OrderedDict

import gensim
import matplotlib.pyplot as plot
import nltk

from mare.requirements import CrowdREReader


class RequirementsPreprocessor(object):

    def __init__(self, path_to_requirements):
        self.crowdre_reader = CrowdREReader(path_to_requirements)
        self.crowdre_reader.read()
        self.requirements_preprocessed = False

    def _preprocess_requirements(self):
        for requirement in self.crowdre_reader.requirements:
            requirement.complete_analysis()
        self.requirements_preprocessed = True

    @property
    def requirements(self):
        if not self.requirements_preprocessed:
            self._preprocess_requirements()
        return self.crowdre_reader.requirements


class RequirementsAnalyzer(object):

    def __init__(self, preprocessed_requirements):
        self.requirements_list = preprocessed_requirements


class NLPAnalyzer(RequirementsAnalyzer):

    def analyze_vocabulary(self):
        # NLP analysis
        joined_text = " ".join(map(lambda re: re.text, self.requirements_list))
        joined_text_tokenized = nltk.word_tokenize(joined_text)

        no_of_requirements = len(self.requirements_list)
        tokens = []
        lexical_words = []
        stems = []
        requirements_with_redundancy = 0

        for requirement in self.requirements_list:
            tokens += requirement.tokens
            lexical_words += requirement.lexical_words
            stems += requirement.stems
            if requirement.contains_redundancy:
                requirements_with_redundancy += 1

        print("Number of Tokens (unique): \t\t{} ({})".format(len(tokens), len(set(tokens))))
        print("Number of Lexical Words: \t\t{}".format(len(lexical_words)))

        print("\nVocabulary Size (Lexical Words): \t{}".format(len(set(lexical_words))))
        print("Vocabulary Size (Stems): \t\t{}".format(len(set(stems))))

        print("\nAverage Sentence Length (Tokens): \t{}".format(round(len(tokens) / no_of_requirements)))
        print("Average Sentence Length (Lexical Words):{}".format(round(len(lexical_words) / no_of_requirements)))

        print("\nLexical Diversity: \t\t\t{}".format(round(len(set(lexical_words)) / len(joined_text), 3)))
        print("Requirements containing\n\t'...I want my smart home to...': \t{}/{} ({}%)".format(
            requirements_with_redundancy, no_of_requirements,
            round(requirements_with_redundancy / no_of_requirements * 100, 2)))

    def analyze_tags(self):
        tags = defaultdict(int)
        tags_per_requirement = []
        tagged_requirements = 0
        for requirement in self.requirements_list:
            for tag in requirement.cleaned_tags:
                tags[tag] += 1
            tags_per_requirement.append(len(requirement.cleaned_tags))
            if len(requirement.cleaned_tags) > 0:
                tagged_requirements += 1

        sorted_tags = OrderedDict(sorted(tags.items(), key=lambda t: t[1], reverse=True))

        x = list(sorted_tags.keys())[:9]
        y = list(sorted_tags.values())[:9]
        plot.bar(x, y)
        plot.suptitle("Tags assigned to requirements and their occurence", fontsize=16)
        plot.show()
        print("Total amount of tags: %d" % sum(tags_per_requirement))
        print("Requirements with tags: {} ({}%)".format(tagged_requirements, round(
            (tagged_requirements / len(self.requirements_list) * 100), 2)))
        print("Tags per Requirement: min: %d, avg: %d, max: %d" % (
            min(tags_per_requirement), sum(tags_per_requirement) / len(tags_per_requirement), max(tags_per_requirement))
              )


class LDAAnalyzer(RequirementsAnalyzer):

    def prepare(self):
        # Prepare Dataset for LDA
        stemsList = []
        for requirement in self.requirements_list:
            requirement.complete_analysis()
            stemsList.append(requirement.stems)

        # Bag of Words on the Data set
        self.dictionary = gensim.corpora.Dictionary(stemsList)

        # Filter out:
        # less than 15 documents (absolute number) or
        # more than 0.5 documents (fraction of total corpus size, not absolute number).
        # after the above two steps, keep only the first 100000 most frequent tokens.
        self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

        # Gensim doc2bow
        self.bow_corpus = [self.dictionary.doc2bow(requirement) for requirement in stemsList]

    def test_preparation(self):
        bow_doc_2965 = self.bow_corpus[2965]
        for i in range(len(bow_doc_2965)):
            print("Word {} (\"{}\") appears {} time.".format(
                bow_doc_2965[i][0], self.dictionary[bow_doc_2965[i][0]], bow_doc_2965[i][1])
            )

    def bag_of_words(self):
        # Running LDA using Bag of Words
        self.bag_of_words_model = gensim.models.LdaMulticore(
            self.bow_corpus, num_topics=10, id2word=self.dictionary, passes=2, workers=2)
        # For each topic, we will explore the words occuring in that topic and its relative weight.
        for idx, topic in self.bag_of_words_model.print_topics(-1):
            print('Topic: {}\nWords: {}\n'.format(idx, topic))

    def tf_idf(self):
        # TF-IDF
        tfidf = gensim.models.TfidfModel(self.bow_corpus)
        corpus_tfidf = tfidf[self.bow_corpus]

        # Running LDA using TF-IDF
        self.tfidf_model = gensim.models.LdaMulticore(
            corpus_tfidf, num_topics=10, id2word=self.dictionary, passes=2, workers=4)
        # For each topic, we will explore the words occuring in that topic and its relative weight.
        for idx, topic in self.tfidf_model.print_topics(-1):
            print('Topic: {}\nWord: {}\n'.format(idx, topic))


class Word2VecAnalyzer(RequirementsAnalyzer):

    def word2vec(self):
        tokenized_sentences = list(map(lambda re: re.tokens, self.requirements_list))
        self.model = gensim.models.Word2Vec(tokenized_sentences, min_count=5, size=50)

#TODO: Word embeddings
