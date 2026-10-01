"""Frozen, local signed hashing features; optional external embeddings may replace them.

This is a transparent lexical baseline, NOT a pretrained semantic embedding model.
"""
from __future__ import annotations
import hashlib
import math
import re


def query_features(text: str, dimensions: int = 64) -> tuple[float,...]:
    if dimensions < 4:raise ValueError("at least four dimensions required")
    values=[0.]*dimensions
    for word in re.findall(r"\w+",text.lower()):
        key=hashlib.sha256(word.encode()).digest()
        values[int.from_bytes(key[:4],"big")%dimensions]+=1. if key[4]&1 else -1.
    norm=math.sqrt(sum(x*x for x in values)) or 1.
    return tuple(x/norm for x in values)
