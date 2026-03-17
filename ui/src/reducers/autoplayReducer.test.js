import { describe, expect, it } from 'vitest'
import { autoplayReducer, DEFAULT_AUTOPLAY_STATE } from './autoplayReducer'
import {
  AUTOPLAY_SET_ENABLED,
  AUTOPLAY_SYNC_SETTINGS,
  AUTOPLAY_TOGGLE_FEEDBACK,
  AUTOPLAY_TRACK_PLAYED,
  AUTOPLAY_TRACKS_REQUESTED,
} from '../actions/autoplay'
import { PLAYER_CLEAR_QUEUE } from '../actions/player'

describe('autoplayReducer', () => {
  it('returns the default state for unknown actions', () => {
    expect(autoplayReducer(undefined, { type: 'UNKNOWN' })).toEqual(
      DEFAULT_AUTOPLAY_STATE,
    )
  })

  it('syncs persisted settings and marks them as loaded', () => {
    const result = autoplayReducer(undefined, {
      type: AUTOPLAY_SYNC_SETTINGS,
      data: {
        enabled: true,
        mode: 'discovery',
        textPrompt: 'late night',
        excludePlaylistIds: ['pl-1'],
        batchSize: 8,
        diversityOverride: 0.7,
      },
    })

    expect(result).toMatchObject({
      enabled: true,
      loaded: true,
      mode: 'discovery',
      textPrompt: 'late night',
      excludePlaylistIds: ['pl-1'],
      batchSize: 8,
      diversityOverride: 0.7,
    })
  })

  it('tracks played/requested ids and clears runtime when disabled', () => {
    let state = autoplayReducer(undefined, {
      type: AUTOPLAY_SET_ENABLED,
      data: { enabled: true },
    })
    state = autoplayReducer(state, {
      type: AUTOPLAY_TRACK_PLAYED,
      data: { trackId: 'song-1' },
    })
    state = autoplayReducer(state, {
      type: AUTOPLAY_TRACKS_REQUESTED,
      data: { trackIds: ['song-2'], source: 'player' },
    })

    expect(state.playedTrackIds).toEqual(['song-1'])
    expect(state.requestedTrackIds).toEqual(['song-2'])
    expect(state.lastRefillSource).toBe('player')

    state = autoplayReducer(state, {
      type: AUTOPLAY_SET_ENABLED,
      data: { enabled: false },
    })

    expect(state.enabled).toBe(false)
    expect(state.playedTrackIds).toEqual([])
    expect(state.requestedTrackIds).toEqual([])
    expect(state.lastRefillSource).toBeNull()
  })

  it('toggles feedback and resets runtime on queue clear', () => {
    let state = autoplayReducer(undefined, {
      type: AUTOPLAY_TOGGLE_FEEDBACK,
      data: { trackId: 'song-1', direction: 'up' },
    })
    state = autoplayReducer(state, {
      type: AUTOPLAY_TOGGLE_FEEDBACK,
      data: { trackId: 'song-2', direction: 'down' },
    })

    expect(state.positiveTrackIds).toEqual(['song-1'])
    expect(state.negativeTrackIds).toEqual(['song-2'])

    state = autoplayReducer(state, { type: PLAYER_CLEAR_QUEUE })

    expect(state.positiveTrackIds).toEqual([])
    expect(state.negativeTrackIds).toEqual([])
    expect(state.fetching).toBe(false)
  })
})
