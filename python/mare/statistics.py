from collections import defaultdict

# Analysis: which tokens can be embedded?


def embeddings_for_tokens(preprocessed_requirements, vectors):
    tokens = []
    for requirement in preprocessed_requirements:
        tokens += requirement.tokens
    unique_tokens = set(tokens)
    print("Unique tokens: ", len(unique_tokens))

    print(
        "In training data: ", len(list(filter(lambda t: t in vectors, unique_tokens))),
    )

    sentences_with_replacements = defaultdict(int)
    for re in preprocessor.requirements:
        unique_tokens = set(re.lexical_words)
        tokens_removed = len(
            list(
                filter(
                    lambda t: not pt_word_2_vec._token_in_training_data(t),
                    unique_tokens,
                )
            )
        )
        sentences_with_replacements[tokens_removed] += 1

    print("\nSentencens with replacements:")
    print("tokens rmvd\tno of reqs")
    for k, v in sentences_with_replacements.items():
        print(k, "\t\t", v)
