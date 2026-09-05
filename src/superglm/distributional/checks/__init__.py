"""Checking, calibration and scoring payloads for a fitted distributional model.

Every module here builds a payload from a fitted ``DenseDistributionalModel``
and returns frozen dataclasses or frames of plain numbers; nothing in this
package draws.  The renderers in :mod:`superglm.plotting` consume the payloads
and never call the model, so a figure is reproducible from its payload alone.
"""
