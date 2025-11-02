"""
Dataset downloaders for various music datasets.
"""

from .base import BaseDownloader
from .song_describer import SongDescriberDownloader
from .jamendo_max_caps import JamendoMaxCapsDownloader
from .mtg_jamendo import MTGJamendoDownloader
from .fma import FMADownloader
from .dali import DALIDownloader
from .clotho import ClothoDownloader

__all__ = [
    'BaseDownloader',
    'SongDescriberDownloader',
    'JamendoMaxCapsDownloader',
    'MTGJamendoDownloader',
    'FMADownloader',
    'DALIDownloader',
    'ClothoDownloader',
]
