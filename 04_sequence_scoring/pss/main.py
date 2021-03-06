#!/usr/bin/env python
#
# Position Specific Scoring (PSS)
#   by Alen Kocaj

from functools import reduce

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch

from pss import PSS
from alphabet import Alphabet

def main():
    # source of aligned sequences being used to
    # build scoring matrix
    source_scoring_file = "testing/source"

    # list of target sequences which should be scored
    target_scoring_file = "testing/target"

    # path where sequence logo of PPM should be saved
    path = "pss_ppm.png"

    # use nucleotide alphabet
    alphabet = Alphabet.nucleotide()
    # the random model used
    weights = {k: v["weight"] for k, v in alphabet.items()}

    # pseudocount for preventing zero probabilities
    pseudocount = 0.0000000001

    print("====================================================================")
    print("Position Specific Scoring (PSS)")
    print("\tby Alen Kocaj")
    print()

    # read source alignments
    print(f"[i] reading source alignments out of '{source_scoring_file}'")
    sources = []
    with open(source_scoring_file) as sourcefm:
        sources = sourcefm.read().split("\n")

    expected_length = len(sources[0]) * len(sources)
    observed_length = reduce(lambda acc, curr: acc + len(curr), sources, 0)

    # validate sources
    avg_length = int(expected_length / len(sources))
    if observed_length != expected_length:
        print(f"[w] not all sequences are of same length. Expected length: {avg_length}")
        print("The overhanging onegram will not be considered.")

    # read target sources
    print(f"[i] reading target sequences out of '{target_scoring_file}'")
    targets = []
    with open(target_scoring_file) as targetfm:
        targets = targetfm.read().split("\n")

    # build matrices
    pss = PSS(sources, alphabet.keys(), weights, avg_length, pseudocount)
    pfm = pss.build_frequency_matrix()
    ppm = pss.build_probability_matrix(pfm)

    print("====================================================================")
    print("[PPM] Position Probability Matrix")
    print(ppm)
    print()
    print("[i] plot sequence logo of PPM")
    plot_ppm_sequence_logo_pd(alphabet, ppm, path, 1.15)
    print(f"[i] figure saved at '{path}'")

    # build Postion Weight Matrix (PWM)
    weight_matrix = pss.build_weight_matrix(ppm)
    print("====================================================================")
    print("[PPM] Probability Weight Matrix")
    print(weight_matrix)

    # score targets
    print("====================================================================")
    print("[i] scoring targets\n")
    for i, target in enumerate(targets):
        print(f"# [{i}th target] #####")
        print(f"> Sequence: {target}")
        print(f"> Log Score by PSS: " + str(pss.score(target, weight_matrix)))
        print()
    print("====================================================================")

    print("[i] thank you and goodnight")

# plot_ppm_sequence_logo_pd plots a Position Probability Matrix (PPM)
# as a Sequence Logo (https://en.wikipedia.org/wiki/Sequence_logo).
# The ppm is given in a Pandas DataFrame format.
# The ppm was built based on a given alphabet.
# Globscale determines the size of each letter within the sequence logo.
#
# Kudos to https://github.com/saketkc
# with https://github.com/saketkc/motif-logos-matplotlib for initial code.
def plot_ppm_sequence_logo_pd(alphabet, ppm, path, custom_y=-1, globscale=1.35):
    fig, ax = plt.subplots(figsize=(10,3))

    x = 1
    maxy = 0
    row_indices = list(ppm.index)
    for column in ppm:

        y = 0
        for row in range(0, len(ppm)):
            score = ppm[column][row]
            base = row_indices[row]

            letter_at(alphabet, base, globscale, x, y, score, ax)
            y += score
        x += 1
        maxy = max(maxy, y)

    plt.xticks(range(1, x))
    plt.xlim((0, x))

    if custom_y != -1:
        maxy = custom_y
    plt.ylim((0, maxy))
    plt.tight_layout()

    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.title(f"Position Probability Matrix of Sequences", y=1.15, fontsize=15)
    plt.suptitle("in " + str.join(", ", row_indices) + " alphabet", y=1.03)
    plt.savefig(f"{path}", bbox_inches="tight")

# letter_at returns the plot element of a given letter from an alphabet
# scaled by globscale and transformed around its x and y axis within the plot
def letter_at(alphabet, letter, globscale, x, y, yscale=1, ax=None):
    text = alphabet[letter]["text"]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=alphabet[letter]["color"], transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

if __name__ == "__main__":
    main()
