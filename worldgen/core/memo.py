"""Values derived from whole preview-grid arrays, computed once per data dict.

The detail window and every block of the 2 m export sample the same grid-wide
fields (slopes, distance transforms, smoothed masks); recomputing them per block
would cost more than the block itself at 2048².
"""


def memo(data, key, make):
    """`make()` once per data dict and key. Keys that depend on an array passed in
    (not taken from `data` by name) include its id()."""
    store = data.setdefault("_memo", {})
    if key not in store:
        store[key] = make()
    return store[key]
