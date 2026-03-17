import {
  SET_NOTIFICATIONS_STATE,
  SET_OMITTED_FIELDS,
  SET_STREAMING_OVERRIDE,
  SET_TOGGLEABLE_FIELDS,
} from '../actions'
import { DEFAULT_STREAMING_OVERRIDE } from '../audioplayer/streamingOverrideUtils'

const initialState = {
  notifications: false,
  toggleableFields: {},
  omittedFields: {},
  streamingOverride: { ...DEFAULT_STREAMING_OVERRIDE },
}

export const settingsReducer = (previousState = initialState, payload) => {
  const { type, data } = payload
  switch (type) {
    case SET_NOTIFICATIONS_STATE:
      return {
        ...previousState,
        notifications: data,
      }
    case SET_TOGGLEABLE_FIELDS:
      return {
        ...previousState,
        toggleableFields: {
          ...previousState.toggleableFields,
          ...data,
        },
      }
    case SET_OMITTED_FIELDS:
      return {
        ...previousState,
        omittedFields: {
          ...previousState.omittedFields,
          ...data,
        },
      }
    case SET_STREAMING_OVERRIDE:
      return {
        ...previousState,
        streamingOverride: {
          ...previousState.streamingOverride,
          ...DEFAULT_STREAMING_OVERRIDE,
          ...data,
        },
      }
    default:
      return previousState
  }
}
