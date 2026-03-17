# Playback Investigation Summary

- Total observations: 57
- GUI observations: 21
- API observations: 36

## Scenario Metrics

| Scenario | Runs | Successes | Failures | Failure % | Median TTFB (ms) | Median Time-to-Play (ms) |
|---|---:|---:|---:|---:|---:|---:|
| API-1 | 9 | 9 | 0 | 0.0 | 1.595 | None |
| API-2 | 9 | 9 | 0 | 0.0 | 1.474 | None |
| API-3 | 9 | 9 | 0 | 0.0 | 1.493 | None |
| API-4 | 9 | 9 | 0 | 0.0 | 2.024 | None |
| GUI-1 | 5 | 0 | 5 | 100.0 | None | None |
| GUI-2 | 5 | 0 | 5 | 100.0 | None | None |
| GUI-3 | 5 | 0 | 5 | 100.0 | None | None |
| GUI-4 | 3 | 0 | 3 | 100.0 | None | None |
| GUI-5 | 3 | 0 | 3 | 100.0 | None | None |

## Ranked Likely Causes

1. **Browser decode/playback issue** (medium) - API stream probes are healthy while GUI FLAC starts frequently fail.
