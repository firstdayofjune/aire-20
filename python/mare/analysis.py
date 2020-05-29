from collections import defaultdict, OrderedDict

import gensim
import matplotlib
import matplotlib.colors as mcolors
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mare.requirements import CrowdREReader
from matplotlib import pyplot as plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# configure matplot to plot with latex font
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams[
    "font.serif"
] = "Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman"


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

        print(
            "Number of Tokens (unique): \t\t{} ({})".format(
                len(tokens), len(set(tokens))
            )
        )
        print("Number of Lexical Words: \t\t{}".format(len(lexical_words)))

        print("\nVocabulary Size (Lexical Words): \t{}".format(len(set(lexical_words))))
        print("Vocabulary Size (Stems): \t\t{}".format(len(set(stems))))

        print(
            "\nAverage Sentence Length (Tokens): \t{}".format(
                round(len(tokens) / no_of_requirements)
            )
        )
        print(
            "Average Sentence Length (Lexical Words):{}".format(
                round(len(lexical_words) / no_of_requirements)
            )
        )

        print(
            "\nLexical Diversity: \t\t\t{}".format(
                round(len(set(lexical_words)) / len(joined_text), 3)
            )
        )
        print(
            "Requirements containing\n\t'...I want my smart home to...': \t{}/{} ({}%)".format(
                requirements_with_redundancy,
                no_of_requirements,
                round(requirements_with_redundancy / no_of_requirements * 100, 2),
            )
        )

    def analyze_tags(self, filename=None):
        tags = defaultdict(int)
        tags_per_requirement = []
        tagged_requirements = 0
        for requirement in self.requirements_list:
            for tag in requirement.cleaned_tags:
                tags[tag] += 1
            tags_per_requirement.append(len(requirement.cleaned_tags))
            if len(requirement.cleaned_tags) > 0:
                tagged_requirements += 1

        sorted_tags = OrderedDict(
            sorted(tags.items(), key=lambda t: t[1], reverse=True)
        )

        x = list(sorted_tags.keys())[:9]
        y = list(sorted_tags.values())[:9]
        figure = plot.figure()
        plot.bar(x, y)
        if filename != None:
            figure.savefig(filename, bbox_inches="tight")
        plot.suptitle("Tags assigned to requirements and their occurence", fontsize=16)
        plot.show()

        print("Total amount of tags: %d" % sum(tags_per_requirement))
        print(
            "Requirements with tags: {} ({}%)".format(
                tagged_requirements,
                round((tagged_requirements / len(self.requirements_list) * 100), 2),
            )
        )
        print(
            "Tags per Requirement: min: %d, avg: %d, max: %d"
            % (
                min(tags_per_requirement),
                sum(tags_per_requirement) / len(tags_per_requirement),
                max(tags_per_requirement),
            )
        )

    def analyze_domains(self, filename=None):
        domainMap = {
            "Energy": 0,
            "Entertainment": 0,
            "Health": 0,
            "Safety": 0,
            "Other": 0,
        }
        for requirement in self.requirements_list:
            domainMap[requirement.domain] = domainMap[requirement.domain] + 1

        sortedDomains = OrderedDict(
            sorted(domainMap.items(), key=lambda t: t[1], reverse=True)
        )

        x = list(sortedDomains.keys())
        y = list(sortedDomains.values())
        print(sortedDomains)
        figure = plot.figure()
        plot.bar(x, y)
        if filename != None:
            figure.savefig(filename, bbox_inches="tight")
        plot.suptitle("Distribution of the domains", fontsize=16)
        plot.show()


class LDAAnalyzer(RequirementsAnalyzer):
    def prepare(self):
        self.num_topics = 5
        # Prepare Dataset for LDA
        self.stemsList = []
        for requirement in self.requirements_list:
            requirement.complete_analysis()
            self.stemsList.append(requirement.stems)

        # Bag of Words on the Data set
        self.dictionary = gensim.corpora.Dictionary(self.stemsList)

        # Filter out:
        # less than 15 documents (absolute number) or
        # more than 0.5 documents (fraction of total corpus size, not absolute number).
        # after the above two steps, keep only the first 100000 most frequent tokens.
        self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

        # Gensim doc2bow
        self.bow_corpus = [
            self.dictionary.doc2bow(requirement) for requirement in self.stemsList
        ]

    def test_preparation(self):
        bow_doc_2965 = self.bow_corpus[2965]
        for i in range(len(bow_doc_2965)):
            print(
                'Word {} ("{}") appears {} time.'.format(
                    bow_doc_2965[i][0],
                    self.dictionary[bow_doc_2965[i][0]],
                    bow_doc_2965[i][1],
                )
            )

    def bag_of_words(self):
        # Running LDA using Bag of Words
        self.bag_of_words_model = gensim.models.LdaMulticore(
            self.bow_corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=2,
            workers=2,
        )
        # For each topic, we will explore the words occuring in that topic and its relative weight.
        # for idx, topic in self.bag_of_words_model.print_topics(-1):
        # print('Topic: {}\nWords: {}\n'.format(idx, topic))

    def tf_idf(self):
        # TF-IDF
        tfidf = gensim.models.TfidfModel(self.bow_corpus)
        self.corpus_tfidf = tfidf[self.bow_corpus]

        # Running LDA using TF-IDF
        self.tfidf_model = gensim.models.LdaMulticore(
            self.corpus_tfidf,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=2,
            workers=4,
        )
        # For each topic, we will explore the words occuring in that topic and its relative weight.
        # for idx, topic in self.tfidf_model.print_topics(-1):
        #    print('Topic: {}\nWord: {}\n'.format(idx, topic))

    def _format_topics_sentences(self, ldamodel=None, corpus=None, texts=None):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series(
                            [int(topic_num), round(prop_topic, 4), topic_keywords]
                        ),
                        ignore_index=True,
                    )
                else:
                    break
        sent_topics_df.columns = [
            "Dominant_Topic",
            "Perc_Contribution",
            "Topic_Keywords",
        ]

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df

    def tf_idf_show(self):
        # Display setting to show more characters in column
        pd.options.display.max_colwidth = 100
        df_topic_sents_keywords = self._format_topics_sentences(
            ldamodel=self.tfidf_model, corpus=self.bow_corpus, texts=self.stemsList
        )
        sent_topics_sorteddf_mallet = pd.DataFrame()
        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby("Dominant_Topic")

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat(
                [
                    sent_topics_sorteddf_mallet,
                    grp.sort_values(["Perc_Contribution"], ascending=False).head(1),
                ],
                axis=0,
            )

        # Reset Index
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

        # Format
        sent_topics_sorteddf_mallet.columns = [
            "Topic_Num",
            "Topic_Perc_Contrib",
            "Keywords",
            "Representative Text",
        ]

        # Show
        display(sent_topics_sorteddf_mallet.head(self.num_topics))

    def visualize_bag_of_words(
        self,
        n_components=2,
        perplexity=30,
        early_exaggeration=12.0,
        learning_rate=100.0,
        verbose=1,
        random_state=0,
        angle=0.99,
        init="pca",
        coloring="domain",
    ):
        self._perform_tsne_visualization(
            self.bag_of_words_model,
            self.bow_corpus,
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            verbose=verbose,
            random_state=random_state,
            angle=angle,
            init=init,
            coloring=coloring,
        )

    def visualize_tf_idf(
        self,
        n_components=2,
        perplexity=30,
        early_exaggeration=12.0,
        learning_rate=100.0,
        verbose=1,
        random_state=0,
        angle=0.99,
        init="pca",
        filename=None,
        coloring="domain",
    ):
        self._perform_tsne_visualization(
            self.tfidf_model,
            self.corpus_tfidf,
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            verbose=verbose,
            random_state=random_state,
            angle=angle,
            init=init,
            filename=filename,
            coloring=coloring,
        )

    def _perform_tsne_visualization(
        self,
        model=None,
        corpus=None,
        n_components=2,
        perplexity=30,
        early_exaggeration=12.0,
        learning_rate=100.0,
        verbose=1,
        random_state=0,
        angle=0.99,
        init="random",
        filename=None,
        coloring="domain",
    ):
        # Prepare colors
        CHERRY = "rgba(137,28,86,.9)"
        TEAL = "rgba(57,117,121,.9)"
        ORANGE = "rgba(212,129,59,.9)"
        PURPLE = "rgba(136,104,156,.9)"
        SAND = "rgba(186,171,155,.9)"

        DOMAIN_COLORS = {
            "Energy": TEAL,
            "Entertainment": SAND,
            "Health": PURPLE,
            "Safety": CHERRY,
            "Other": ORANGE,
        }
        #           '6': PURPLE,
        #    '7': "rgb(155,216,153)",

        # Prepare the color list for plotting tags with colors
        color_list = []
        tag_list = []
        for requirement in self.requirements_list:
            tag_found = False
            for domain in DOMAIN_COLORS:
                if domain in requirement.domain:
                    # append the matching color if found
                    color_list.append(DOMAIN_COLORS.get(domain))
                    tag_list.append(str(domain))
                    tag_found = True
                    break
            if not tag_found:
                # If not in the top list show them white
                color_list.append("rgb(220,220,220)")
                tag_list.append("Not Found")
        # Get topic weights
        topic_weights = []
        for i, row_list in enumerate(model[corpus]):
            sentence = []
            for item in row_list:
                sentence.append(item[1])
            topic_weights.append(sentence)
        # Array of topic weights
        arr = pd.DataFrame(topic_weights).fillna(0).values

        # Dominant topic number in each doc
        topic_num = np.argmax(arr, axis=1)
        # tSNE Dimension Reduction
        tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            verbose=verbose,
            random_state=random_state,
            angle=angle,
            init=init,
        )
        tsne_lda = tsne_model.fit_transform(arr)

        fig = go.Figure()
        if coloring == "domains":
            tsne_data = pd.DataFrame(
                {
                    "x": tsne_lda[:, 0],
                    "y": tsne_lda[:, 1],
                    "groups": tag_list,
                    "colors": color_list,
                }
            )
            for tag in list(DOMAIN_COLORS.keys()):
                tag_data = tsne_data[tsne_data["groups"] == tag]
                fig.add_trace(
                    go.Scatter(
                        x=tag_data["x"],
                        y=tag_data["y"],
                        name=tag,
                        mode="markers",
                        marker_color=tag_data["colors"],
                    )
                )
        elif coloring == "topics":
            tsne_data = pd.DataFrame(
                {"x": tsne_lda[:, 0], "y": tsne_lda[:, 1], "groups": topic_num}
            )
            topic_colors = list(DOMAIN_COLORS.values())
            for topicNumber in range(len(topic_colors)):
                tag_data = tsne_data[tsne_data["groups"] == topicNumber]
                fig.add_trace(
                    go.Scatter(
                        x=tag_data["x"],
                        y=tag_data["y"],
                        name=topicNumber,
                        mode="markers",
                        marker_color=topic_colors[topicNumber],
                    )
                )
        # Plot or save as file
        if filename == None:
            fig.show()
        else:
            fig.write_image(filename)
            fig.show()

    def _perform_tsne_visualization_tag(
        self,
        model=None,
        corpus=None,
        n_components=2,
        perplexity=30,
        early_exaggeration=12.0,
        learning_rate=100.0,
        verbose=1,
        random_state=0,
        angle=0.99,
        init="random",
    ):
        # Get most popular tags
        tags = defaultdict(int)
        tags_per_requirement = []
        tagged_requirements = 0
        for requirement in self.requirements_list:
            for tag in requirement.cleaned_tags:
                tags[tag] += 1
            tags_per_requirement.append(len(requirement.cleaned_tags))
            if len(requirement.cleaned_tags) > 0:
                tagged_requirements += 1

        sorted_tags = OrderedDict(
            sorted(tags.items(), key=lambda t: t[1], reverse=True)
        )
        topic_list = list(sorted_tags.keys())[: self.num_topics]
        # Prepare colors
        prepared_colors = np.array(
            [color for name, color in mcolors.TABLEAU_COLORS.items()]
        )
        DOMAIN_COLORS = dict(zip(topic_list, prepared_colors[: self.num_topics]))
        # Prepare the color list for plotting tags with colors
        color_list = []
        tag_list = []
        for requirement in self.requirements_list:
            tag_found = False
            for tag in DOMAIN_COLORS:
                if tag in requirement.cleaned_tags:
                    # append the matching color if found
                    color_list.append(DOMAIN_COLORS.get(tag))
                    tag_list.append(str(tag))
                    tag_found = True
                    break
            if not tag_found:
                # If not in the top list show them white
                raise ("Not matching")
                color_list.append("rgb(220,220,220)")
                tag_list.append("other")
        # Get topic weights
        topic_weights = []
        for i, row_list in enumerate(model[corpus]):
            sentence = []
            for item in row_list:
                sentence.append(item[1])
            topic_weights.append(sentence)
        # Array of topic weights
        arr = pd.DataFrame(topic_weights).fillna(0).values

        # Dominant topic number in each doc
        topic_num = np.argmax(arr, axis=1)

        # tSNE Dimension Reduction
        tsne_model = TSNE(
            n_components=n_components,
            verbose=verbose,
            random_state=random_state,
            angle=angle,
            init=init,
        )
        tsne_lda = tsne_model.fit_transform(arr)

        fig = go.Figure()
        tsne_data = pd.DataFrame(
            {
                "x": tsne_lda[:, 0],
                "y": tsne_lda[:, 1],
                "groups": tag_list,
                "colors": color_list,
            }
        )
        topic_list.append("other")
        for tag in topic_list:
            tag_data = tsne_data[tsne_data["groups"] == tag]
            fig.add_trace(
                go.Scatter(
                    x=tag_data["x"],
                    y=tag_data["y"],
                    name=tag,
                    mode="markers",
                    marker_color=tag_data["colors"],
                )
            )
        fig.show()


class Word2VecAnalyzer(RequirementsAnalyzer):
    def __init__(self, preprocessed_requirements):
        super().__init__(preprocessed_requirements)

        self.vectors = []
        self.requirement_vectors = []
        # sentences are used for plot annotations
        self.sentences = []
        # domains are used to color the plot
        self.domains = []
        # the shortest_req is needed to reduce dimensions
        self.shortest_req = 0
        self.longest_req = 0
        self.traces = []

    def _token_not_redundant(self, token):
        return token.lower() not in [
            "as",
            "smart",
            "home",
            "owner",
            "i",
            "want",
            "be",
            "able",
        ]

    def _token_in_training_data(self, token):
        return token in self.vectors

    def _filter_tokens(self, tokens):
        return filter(
            lambda token: self._token_not_redundant(token)
            and self._token_in_training_data(token),
            tokens,
        )

    def build_requirement_vectors(self, threshold=0, force_overwrite=False):
        """Replaces each word in a given list of RE sentences with its vector representation."""
        if self.requirement_vectors and not force_overwrite:
            raise Exception(
                "Requirement vectors already exists. To overwrite the existing data set force_overwrite=True."
            )
        self.requirement_vectors = []
        self.shortest_req = float("inf")
        self.longest_req = 0
        self.sentences = []
        self.domains = []

        for requirement in self.requirements_list:
            # the lexical words are the tokenized sentences which were freed from stopwords
            filtered_tokens = list(self._filter_tokens(requirement.lexical_words))
            if len(filtered_tokens) == 0:
                print(
                    "Sentence cannot be embedded:\n\t",
                    requirement,
                    requirement.lexical_words,
                )
                continue
            sentence = self.vectors[filtered_tokens]
            if len(sentence) >= threshold:
                self.shortest_req = min(len(sentence), self.shortest_req)
                self.longest_req = max(len(sentence), self.longest_req)
                self.requirement_vectors.append(sentence.transpose())
                self.sentences.append(requirement.cleaned_text)
                self.domains.append(requirement.domain)

    def reduce_dimensions(self):
        pcaed = []
        pca = PCA(n_components=self.shortest_req)
        for vector in self.requirement_vectors:
            pcaed.append(pca.fit_transform(vector))
        return np.array(pcaed)

    def tsne_traces(self, data, perplexity=50.0, learning_rate=200.0):
        self.traces = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_jobs=-1,
        ).fit_transform(data)

    def save_traces(self, path):
        np.savetxt(path, self.traces, delimiter=";")


class PreTrainedWord2VecAnalyzer(Word2VecAnalyzer):
    def load(self, path):
        self.vectors = gensim.models.KeyedVectors.load_word2vec_format(
            path, binary=True
        )


class SelfTrainedWord2VecAnalyzer(Word2VecAnalyzer):
    # Training algorithms
    CONTINUOUS_BAG_OF_WORDS = 0
    SKIP_GRAM = 1

    def train(self, min_occurrences=5, layers=50, training_algorithm=0):
        sentences = list(map(lambda re: re.tokens, self.requirements_list))
        model = gensim.models.Word2Vec(
            sentences, min_count=min_occurrences, size=layers, sg=training_algorithm
        )
        self.vectors = model.wv
