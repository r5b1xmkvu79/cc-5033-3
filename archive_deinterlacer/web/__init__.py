"""Flask web layer.

This package contains no deinterlacing math: it uploads clips, calls the
core engine and renders its diagnostics, fields, motion masks and
reconstructed frames. When the core refuses an input the message is shown in
the UI rather than papered over with a guess.
"""
