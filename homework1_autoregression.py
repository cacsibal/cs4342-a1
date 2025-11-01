from collections import Counter
from datasets import load_dataset
import numpy as np

# Load dataset of children's stories
dataset = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=42)
counter = Counter()  # Keep track of most common words
ENOUGH_EXAMPLES = 100000
for i, story in enumerate(dataset):
    if i == ENOUGH_EXAMPLES:
        break
    # Ignore case and punctuation such as commas and quotation marks.
    words = story['text'].upper().replace(",", "").replace("\n", " ").replace('"', '').replace("!", " ").replace(".", " ").split(' ')
    filteredWords = [ w for w in words if w != "" ]
    counter.update(filteredWords)

# Select the most common words
NUM_WORDS = 501
topWords = [ w[0] for w in counter.most_common(NUM_WORDS-1) ]  # leave room for "."
topWords.append(".")  # end-of-sentence symbol "."
wordToIdxMap = { topWords[i]:i for i in range(NUM_WORDS) }  # map from a word to its index in the topWords list

# TASK 1 (training): estimate the three probability distributions P(x_1), P(x_2 | x_1), and P(x_{t+2} | x_t, x_{t+1}).
# TODO: initialize np.array's to represent the probability distributions

# finding P(x_1)
first_word_counts = np.zeros(NUM_WORDS)

for i, story in enumerate(dataset):
    if i == ENOUGH_EXAMPLES:
        break

    words = story['text'].upper().replace(",", "").replace("\n", " ").replace('"', '').replace("!", " ").replace(".", " ").split(' ')
    filteredWords = [w for w in words if w != ""]

    if len(filteredWords) > 0:
        first_word = filteredWords[0]

        if first_word in wordToIdxMap:
            idx = wordToIdxMap[first_word]
        else:
            idx = NUM_WORDS - 1

        first_word_counts[idx] += 1

P_x1 = first_word_counts / first_word_counts.sum()

# print(P_x1)
# print(np.sum(P_x1))

# finding P(x_2 | x_1)
# bigram_counts[i, j] = count of (x_1=word_i, x_2=word_j)
bigram_counts = np.zeros((NUM_WORDS, NUM_WORDS))

for i, story in enumerate(dataset):
    if i == ENOUGH_EXAMPLES:
        break

    words = story['text'].upper().replace(",", "").replace("\n", " ").replace('"', '').replace("!", " ").replace(".", " ").split(' ')
    filteredWords = [w for w in words if w != ""]

    if len(filteredWords) >= 2:
        first_word = filteredWords[0]
        second_word = filteredWords[1]

        if first_word in wordToIdxMap:
            idx1 = wordToIdxMap[first_word]
        else:
            idx1 = NUM_WORDS - 1

        if second_word in wordToIdxMap:
            idx2 = wordToIdxMap[second_word]
        else:
            idx2 = NUM_WORDS - 1

        bigram_counts[idx1, idx2] += 1

    P_x2_given_x1 = np.zeros((NUM_WORDS, NUM_WORDS))
for i in range(NUM_WORDS):
    row_sum = bigram_counts[i].sum()
    if row_sum > 0:
        P_x2_given_x1[i] = bigram_counts[i] / row_sum

# print(P_x2_given_x1.shape)
# print(P_x2_given_x1.sum(axis=1)[:10])

trigram_counts = np.zeros((NUM_WORDS, NUM_WORDS, NUM_WORDS))

for i, story in enumerate(dataset):
    if i == ENOUGH_EXAMPLES:
        break
    # Split each story into sentences, ignoring case and punctuation.
    sentences = story['text'].upper().replace(",", "").replace("\n", " ").replace('"', '').replace("!", ".").split('. ')
    for sentence in sentences:
        # Convert each sentence into a word sequence
        sentence = sentence.replace(".", "")
        words = sentence.split(" ") + ["."]
        if not set(words).issubset(topWords):  # Ignore any sentence that contains ineligible words
            continue
        # TODO: process all the 3-grams in each sentence, but ignore any 3-gram if it contains 
        # a word that is not in topWords.
       

        #Filter out empty strings
        filteredWords = [ w for w in words if w != "" ]

        #Process all trigrams in sentence
        for t in range(len(filteredWords) - 2):
            word_t = filteredWords[t]
            word_t1 = filteredWords[t + 1]
            word_t2 = filteredWords[t + 2]

            idx1 = wordToIdxMap[word_t]
            idx2 = wordToIdxMap[word_t1]
            idx3 = wordToIdxMap[word_t2]

            # Increment the trigram count
            trigram_counts[idx1, idx2, idx3] += 1


# TODO: normalize the probability distributions.
P_xt2_given_xt_xt1 = np.zeros((NUM_WORDS, NUM_WORDS, NUM_WORDS))
for i in range(NUM_WORDS):
    for j in range(NUM_WORDS):
        context_sum = trigram_counts[i, j, :].sum()
        if context_sum > 0:
            P_xt2_given_xt_xt1[i, j] = trigram_counts[i, j] / context_sum
            
# TASK 2 (testing/inference): use the probability distributions to generate 100 new "sentences".
# Note: given this relatively weak 3-gram model, not all sentences will be grammatical.
# This is ok for this assignment.
# To select from any probability distribution, you can use np.random.choice.
# TODO ...

print("done training:")
for i in range(100):
    sentence = []

    x_1_index = -1
    x_1 = "."
    while x_1 == ".":
        x_1_index = np.random.choice(range(NUM_WORDS), p=P_x1)
        x_1 = topWords[x_1_index]
    sentence.append(x_1)

    x_2_index = -1
    x_2 = "."
    while x_2 == ".":
        x_2_index = np.random.choice(range(NUM_WORDS), p=P_x2_given_x1[x_1_index, :])
        x_2 = topWords[x_2_index]
    sentence.append(x_2)

    prev_prev_index = x_1_index
    prev_index = x_2_index

    while True:
        distribution = P_xt2_given_xt_xt1[prev_prev_index, prev_index, :]
        if distribution.sum() == 0:
            distribution = np.ones(NUM_WORDS) / NUM_WORDS

        next_index = np.random.choice(range(NUM_WORDS), p=distribution)
        next_word = topWords[next_index]
        sentence.append(next_word)

        if next_word == ".":
            break

        prev_prev_index = prev_index
        prev_index = next_index

    print(f"{i + 1}: {' '.join(sentence)}")

most_common_first_word_index = np.argmax(first_word_counts)
most_common_first_word = topWords[most_common_first_word_index]
count = first_word_counts[most_common_first_word_index]

print(f"Most common first word: '{most_common_first_word}' with {count} occurrences")
print(f"Probability: {P_x1[most_common_first_word_index]:.4f}")