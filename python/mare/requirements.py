import csv
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Requirement(object):
    def __init__(self, text, tags, domain):
        self.text = text
        self.cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)
        self.tags = tags.split(",")
        self.cleaned_tags = self.clean_tags(self.tags)
        self.domain = domain

    def tokenize(self):
        self.tokens = nltk.word_tokenize(self.cleaned_text)

    def check_redundancy(self):
        self.contains_redundancy = "i want my smart home to" in self.text.lower()

    def remove_stopwords(self):
        self.lexical_words = [word for word in self.tokens if word not in stopwords.words('english')]

    def reduce_to_stem(self):
        # lancasterStemmer = LancasterStemmer()
        porter_stemmer = PorterStemmer()
        self.stems = []
        for lexicalWord in self.lexical_words:
            self.stems.append(porter_stemmer.stem(lexicalWord))

    def clean_tags(self, tags):
        whitespace_filter = lambda t: t != " " and t != ""
        whitespace_strip = lambda t: t.strip()
        cleaned_tags = list(map(whitespace_strip, filter(whitespace_filter, tags)))
        return cleaned_tags

    def complete_analysis(self):
        self.tokenize()
        self.check_redundancy()
        self.remove_stopwords()
        self.reduce_to_stem()

    def __str__(self):
        return self.text


class CrowdREReader(object):
    def __init__(self, path):
        if not path.exists() and path.is_file():
            raise ("The given path does not exist or is not a file.")
        self.csv = path
        self.requirements = []

    def read(self):
        with open(self.csv, newline='') as requirements_csv:
            re_reader = csv.DictReader(requirements_csv, delimiter=',')
            for row in re_reader:
                requirement = self._read_row(row)
                self.requirements.append(requirement)

    def _read_row(self, row):
        role = row['role']
        feature = row['feature']
        benefit = row['benefit']
        tags = row['tags']
        domain = row['application_domain']

        requirement_text = self._build_requirement_text(role, feature, benefit)
        return Requirement(requirement_text, tags, domain)

    def _build_requirement_text(self, role, feature, benefit):
        requirementText = "As a {role} I want {feature} so that {benefit}".format(
            role=role, feature=feature, benefit=benefit
        )
        return requirementText
