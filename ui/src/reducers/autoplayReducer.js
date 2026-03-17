import {
  AUTOPLAY_FETCH_FINISH,
  AUTOPLAY_FETCH_START,
  AUTOPLAY_RESET_RUNTIME,
  AUTOPLAY_SET_ENABLED,
  AUTOPLAY_SYNC_SETTINGS,
  AUTOPLAY_TOGGLE_FEEDBACK,
  AUTOPLAY_TRACK_PLAYED,
  AUTOPLAY_TRACKS_REQUESTED,
} from '../actions/autoplay'
import { PLAYER_CLEAR_QUEUE } from '../actions/player'
import { computeFeedback, ensureUniqueFeedback } from '../autoplay/feedbackUtils'

const HISTORY_LIMIT = 500

export const DEFAULT_AUTOPLAY_STATE = {
  enabled: false,
  mode: 'recent',
  textPrompt: '',
  excludePlaylistIds: [],
  batchSize: 5,
  diversityOverride: null,
  loaded: false,
  fetching: false,
  playedTrackIds: [],
  requestedTrackIds: [],
  positiveTrackIds: [],
  negativeTrackIds: [],
  lastRefillSource: null,
}

const normalizeTrackId = (trackId) => {
  if (trackId === null || trackId === undefined) {
    return ''
  }
  return trackId.toString().trim()
}

const uniqueNonEmptyStrings = (values = []) => {
  const result = []
  const seen = new Set()
  values.forEach((value) => {
    const normalized = normalizeTrackId(value)
    if (!normalized || seen.has(normalized)) {
      return
    }
    seen.add(normalized)
    result.push(normalized)
  })
  return result
}

const appendHistory = (values, nextValues) => {
  const merged = uniqueNonEmptyStrings([...(values || []), ...(nextValues || [])])
  if (merged.length <= HISTORY_LIMIT) {
    return merged
  }
  return merged.slice(merged.length - HISTORY_LIMIT)
}

const runtimeDefaults = {
  fetching: false,
  playedTrackIds: [],
  requestedTrackIds: [],
  positiveTrackIds: [],
  negativeTrackIds: [],
  lastRefillSource: null,
}

const normalizeSettings = (data = {}, previousState = DEFAULT_AUTOPLAY_STATE) => {
  const feedback = ensureUniqueFeedback(
    data.positiveTrackIds ?? previousState.positiveTrackIds,
    data.negativeTrackIds ?? previousState.negativeTrackIds,
  )

  return {
    enabled:
      typeof data.enabled === 'boolean'
        ? data.enabled
        : previousState.enabled ?? DEFAULT_AUTOPLAY_STATE.enabled,
    mode:
      typeof data.mode === 'string' && data.mode.trim()
        ? data.mode.trim().toLowerCase()
        : previousState.mode ?? DEFAULT_AUTOPLAY_STATE.mode,
    textPrompt:
      typeof data.textPrompt === 'string'
        ? data.textPrompt
        : previousState.textPrompt ?? DEFAULT_AUTOPLAY_STATE.textPrompt,
    excludePlaylistIds: uniqueNonEmptyStrings(
      Array.isArray(data.excludePlaylistIds)
        ? data.excludePlaylistIds
        : previousState.excludePlaylistIds,
    ),
    batchSize:
      Number.isFinite(Number(data.batchSize)) && Number(data.batchSize) > 0
        ? Number(data.batchSize)
        : previousState.batchSize ?? DEFAULT_AUTOPLAY_STATE.batchSize,
    diversityOverride:
      data.diversityOverride === null || data.diversityOverride === undefined
        ? null
        : Number(data.diversityOverride),
    loaded:
      typeof data.loaded === 'boolean'
        ? data.loaded
        : previousState.loaded ?? DEFAULT_AUTOPLAY_STATE.loaded,
    fetching:
      typeof data.fetching === 'boolean'
        ? data.fetching
        : previousState.fetching ?? DEFAULT_AUTOPLAY_STATE.fetching,
    playedTrackIds: appendHistory(
      [],
      Array.isArray(data.playedTrackIds)
        ? data.playedTrackIds
        : previousState.playedTrackIds,
    ),
    requestedTrackIds: appendHistory(
      [],
      Array.isArray(data.requestedTrackIds)
        ? data.requestedTrackIds
        : previousState.requestedTrackIds,
    ),
    positiveTrackIds: feedback.positive,
    negativeTrackIds: feedback.negative,
    lastRefillSource:
      typeof data.lastRefillSource === 'string' && data.lastRefillSource.trim()
        ? data.lastRefillSource
        : previousState.lastRefillSource ?? DEFAULT_AUTOPLAY_STATE.lastRefillSource,
  }
}

const resetRuntime = (state) => ({
  ...state,
  ...runtimeDefaults,
})

export const autoplayReducer = (
  previousState = DEFAULT_AUTOPLAY_STATE,
  payload,
) => {
  const { type, data } = payload

  switch (type) {
    case AUTOPLAY_SYNC_SETTINGS: {
      const normalized = normalizeSettings(
        { ...data, loaded: true, fetching: false },
        previousState,
      )
      return normalized.enabled ? normalized : resetRuntime(normalized)
    }
    case AUTOPLAY_SET_ENABLED:
      if (data?.enabled) {
        return { ...previousState, enabled: true, loaded: true }
      }
      return resetRuntime({ ...previousState, enabled: false, loaded: true })
    case AUTOPLAY_FETCH_START:
      return {
        ...previousState,
        fetching: true,
      }
    case AUTOPLAY_FETCH_FINISH:
      return {
        ...previousState,
        fetching: false,
      }
    case AUTOPLAY_TRACK_PLAYED: {
      const trackId = normalizeTrackId(data?.trackId)
      if (!trackId) {
        return previousState
      }
      return {
        ...previousState,
        playedTrackIds: appendHistory(previousState.playedTrackIds, [trackId]),
      }
    }
    case AUTOPLAY_TRACKS_REQUESTED:
      return {
        ...previousState,
        requestedTrackIds: appendHistory(
          previousState.requestedTrackIds,
          data?.trackIds,
        ),
        lastRefillSource:
          typeof data?.source === 'string' && data.source.trim()
            ? data.source
            : previousState.lastRefillSource,
      }
    case AUTOPLAY_TOGGLE_FEEDBACK: {
      const result = computeFeedback(
        previousState.positiveTrackIds,
        previousState.negativeTrackIds,
        data?.trackId,
        data?.direction,
      )
      return {
        ...previousState,
        positiveTrackIds: result.positive,
        negativeTrackIds: result.negative,
      }
    }
    case AUTOPLAY_RESET_RUNTIME:
    case PLAYER_CLEAR_QUEUE:
      return resetRuntime(previousState)
    default:
      return previousState
  }
}
