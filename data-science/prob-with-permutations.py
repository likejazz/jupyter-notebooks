# %%
from random import randint


def gen_str(length=10):
    alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    str = []
    for i in range(length):
        str.append(alphanum[randint(0, len(alphanum) - 1)])

    return ''.join(str)


sentence = gen_str(10)

total_trials = 1000000
pattern_length = 3
matched_count = 0
for i in range(total_trials):
    pattern = gen_str(pattern_length)
    if pattern in sentence:
        matched_count += 1

# Print out Permutations with repetition probability.
print("%d/%d %f" % (matched_count, total_trials, matched_count / total_trials))
