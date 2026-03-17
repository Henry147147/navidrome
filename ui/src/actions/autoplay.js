export const AUTOPLAY_SYNC_SETTINGS = 'AUTOPLAY_SYNC_SETTINGS'
export const AUTOPLAY_SET_ENABLED = 'AUTOPLAY_SET_ENABLED'
export const AUTOPLAY_FETCH_START = 'AUTOPLAY_FETCH_START'
export const AUTOPLAY_FETCH_FINISH = 'AUTOPLAY_FETCH_FINISH'
export const AUTOPLAY_TRACK_PLAYED = 'AUTOPLAY_TRACK_PLAYED'
export const AUTOPLAY_TRACKS_REQUESTED = 'AUTOPLAY_TRACKS_REQUESTED'
export const AUTOPLAY_TOGGLE_FEEDBACK = 'AUTOPLAY_TOGGLE_FEEDBACK'
export const AUTOPLAY_RESET_RUNTIME = 'AUTOPLAY_RESET_RUNTIME'

export const syncAutoPlaySettings = (settings) => ({
  type: AUTOPLAY_SYNC_SETTINGS,
  data: settings,
})

export const setAutoPlayEnabled = (enabled) => ({
  type: AUTOPLAY_SET_ENABLED,
  data: { enabled },
})

export const startAutoPlayFetch = () => ({
  type: AUTOPLAY_FETCH_START,
})

export const finishAutoPlayFetch = () => ({
  type: AUTOPLAY_FETCH_FINISH,
})

export const markAutoPlayTrackPlayed = (trackId) => ({
  type: AUTOPLAY_TRACK_PLAYED,
  data: { trackId },
})

export const markAutoPlayTracksRequested = (trackIds, source) => ({
  type: AUTOPLAY_TRACKS_REQUESTED,
  data: { trackIds, source },
})

export const toggleAutoPlayFeedback = (trackId, direction) => ({
  type: AUTOPLAY_TOGGLE_FEEDBACK,
  data: { trackId, direction },
})

export const resetAutoPlayRuntime = () => ({
  type: AUTOPLAY_RESET_RUNTIME,
})
