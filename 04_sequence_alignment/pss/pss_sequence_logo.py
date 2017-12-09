#!/usr/bin/env python
#
# Kudos to https://github.com/saketkc 
# with https://github.com/saketkc/motif-logos-matplotlib

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

globscale = 1.35
LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1),
            "G" : TextPath((-0.384, 0), "G", size=1),
            "A" : TextPath((-0.35, 0), "A", size=1),
            "C" : TextPath((-0.366, 0), "C", size=1) }
COLOR_SCHEME = {'G': 'orange', 
                'A': 'red', 
                'C': 'blue', 
                'T': 'darkgreen'}

def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p