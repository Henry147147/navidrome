import { describe, expect, it } from 'vitest'
import { settingsReducer } from './settingsReducer'
import {
  SET_NOTIFICATIONS_STATE,
  SET_OMITTED_FIELDS,
  SET_STREAMING_OVERRIDE,
  SET_TOGGLEABLE_FIELDS,
} from '../actions'

describe('settingsReducer', () => {
  it('returns initial state when action is unknown', () => {
    const result = settingsReducer(undefined, { type: 'UNKNOWN_ACTION' })
    expect(result).toEqual({
      notifications: false,
      toggleableFields: {},
      omittedFields: {},
      streamingOverride: {
        mode: 'default',
        profileId: null,
        format: null,
        maxBitRate: null,
      },
    })
  })

  it('updates streaming override payload', () => {
    const previousState = settingsReducer(undefined, { type: 'INIT' })
    const action = {
      type: SET_STREAMING_OVERRIDE,
      data: {
        mode: 'override',
        profileId: 'tr_opus',
        format: 'opus',
        maxBitRate: 192,
      },
    }

    const result = settingsReducer(previousState, action)
    expect(result.streamingOverride).toEqual({
      mode: 'override',
      profileId: 'tr_opus',
      format: 'opus',
      maxBitRate: 192,
    })
  })

  it('resets override fields when default mode payload is dispatched', () => {
    const previousState = {
      notifications: true,
      toggleableFields: { song: ['album'] },
      omittedFields: { song: ['comment'] },
      streamingOverride: {
        mode: 'override',
        profileId: 'tr_opus',
        format: 'opus',
        maxBitRate: 192,
      },
    }
    const action = {
      type: SET_STREAMING_OVERRIDE,
      data: {
        mode: 'default',
      },
    }

    const result = settingsReducer(previousState, action)
    expect(result.streamingOverride).toEqual({
      mode: 'default',
      profileId: null,
      format: null,
      maxBitRate: null,
    })
    expect(result.notifications).toBe(true)
    expect(result.toggleableFields).toEqual({ song: ['album'] })
    expect(result.omittedFields).toEqual({ song: ['comment'] })
  })

  it('keeps existing behavior for other settings actions', () => {
    const previousState = settingsReducer(undefined, { type: 'INIT' })
    const notifications = settingsReducer(previousState, {
      type: SET_NOTIFICATIONS_STATE,
      data: true,
    })
    expect(notifications.notifications).toBe(true)

    const toggled = settingsReducer(notifications, {
      type: SET_TOGGLEABLE_FIELDS,
      data: { song: ['album'] },
    })
    expect(toggled.toggleableFields).toEqual({ song: ['album'] })

    const omitted = settingsReducer(toggled, {
      type: SET_OMITTED_FIELDS,
      data: { song: ['comment'] },
    })
    expect(omitted.omittedFields).toEqual({ song: ['comment'] })
  })
})
