import {
  addTracks,
  appendSongs,
  finishAutoPlayFetch,
  markAutoPlayTracksRequested,
  playTracks,
  startAutoPlayFetch,
} from '../actions'
import {
  formatRecommendationError,
  sanitizeRecommendationWarning,
} from '../recommendationHealth'

export const normalizeAutoPlaySettings = (data = {}) => ({
  enabled: Boolean(data?.enabled),
  mode:
    typeof data?.mode === 'string' && data.mode.trim()
      ? data.mode.trim().toLowerCase()
      : 'recent',
  textPrompt: typeof data?.textPrompt === 'string' ? data.textPrompt : '',
  excludePlaylistIds: Array.isArray(data?.excludePlaylistIds)
    ? data.excludePlaylistIds.filter(Boolean)
    : [],
  batchSize:
    Number.isFinite(Number(data?.batchSize)) && Number(data.batchSize) > 0
      ? Number(data.batchSize)
      : 5,
  diversityOverride:
    data?.diversityOverride === null || data?.diversityOverride === undefined
      ? null
      : Number(data.diversityOverride),
})

export const getQueueItemTrackId = (item) => {
  if (!item) {
    return ''
  }
  const trackId = item.trackId || item.song?.id || item.song?.mediaFileId
  if (trackId === null || trackId === undefined) {
    return ''
  }
  return trackId.toString().trim()
}

export const getPlayerCurrentIndex = (player) => {
  const queue = player?.queue || []
  const currentUuid = player?.current?.uuid
  if (!currentUuid) {
    return queue.length > 0 ? 0 : -1
  }
  return queue.findIndex((item) => item.uuid === currentUuid)
}

export const getRemainingQueue = (player) => {
  const queue = player?.queue || []
  if (queue.length === 0) {
    return 0
  }
  const currentIndex = getPlayerCurrentIndex(player)
  if (currentIndex < 0) {
    return queue.length
  }
  return Math.max(queue.length - currentIndex - 1, 0)
}

const uniqueNonEmpty = (values = []) => {
  const result = []
  const seen = new Set()
  values.forEach((value) => {
    const normalized =
      value === null || value === undefined ? '' : value.toString().trim()
    if (!normalized || seen.has(normalized)) {
      return
    }
    seen.add(normalized)
    result.push(normalized)
  })
  return result
}

const getNonRadioQueueItems = (queue = []) =>
  queue.filter((item) => !item?.isRadio && getQueueItemTrackId(item))

const getQueueTailTrackIds = (player) => {
  const queue = player?.queue || []
  const currentIndex = getPlayerCurrentIndex(player)
  const tail = currentIndex >= 0 ? queue.slice(currentIndex) : queue
  return uniqueNonEmpty(tail.map((item) => getQueueItemTrackId(item)))
}

const getAllQueueTrackIds = (player) =>
  uniqueNonEmpty((player?.queue || []).map((item) => getQueueItemTrackId(item)))

export const buildQueueSeedTrackIds = (player, autoplay, limit = 5) => {
  const queue = getNonRadioQueueItems(player?.queue || [])
  const currentIndex = getPlayerCurrentIndex(player)
  const sourceItems =
    currentIndex >= 0
      ? queue.slice(0, currentIndex + 1).reverse()
      : [...queue].reverse()
  const seedIds = []
  const seen = new Set()

  sourceItems.forEach((item) => {
    if (seedIds.length >= limit) {
      return
    }
    const trackId = getQueueItemTrackId(item)
    if (!trackId || seen.has(trackId)) {
      return
    }
    seen.add(trackId)
    seedIds.push(trackId)
  })

  const playedTrackIds = [...(autoplay?.playedTrackIds || [])].reverse()
  playedTrackIds.forEach((trackId) => {
    if (seedIds.length >= limit) {
      return
    }
    const normalized = trackId?.toString().trim()
    if (!normalized || seen.has(normalized)) {
      return
    }
    seen.add(normalized)
    seedIds.push(normalized)
  })

  return seedIds
}

export const buildExcludeTrackIds = (player, autoplay) =>
  uniqueNonEmpty([
    ...(autoplay?.playedTrackIds || []),
    ...(autoplay?.requestedTrackIds || []),
    ...(autoplay?.negativeTrackIds || []),
    ...getQueueTailTrackIds(player),
  ])

const buildPayloadBase = (player, autoplay) => {
  const payload = {
    limit: autoplay?.batchSize || 5,
    excludeTrackIds: buildExcludeTrackIds(player, autoplay),
    excludePlaylistIds: autoplay?.excludePlaylistIds || [],
    positiveTrackIds: autoplay?.positiveTrackIds || [],
    negativeTrackIds: autoplay?.negativeTrackIds || [],
  }
  if (
    autoplay?.diversityOverride !== null &&
    autoplay?.diversityOverride !== undefined &&
    autoplay?.diversityOverride !== ''
  ) {
    payload.diversity = Number(autoplay.diversityOverride)
  }
  return payload
}

const buildFallbackRequest = (dataProvider, autoplay, payloadBase) => {
  const mode = autoplay?.mode
  switch (mode) {
    case 'recent':
      return {
        source: 'recent',
        run: () => dataProvider.getRecentRecommendations(payloadBase),
      }
    case 'favorites':
      return {
        source: 'favorites',
        run: () => dataProvider.getFavoriteRecommendations(payloadBase),
      }
    case 'all':
      return {
        source: 'all',
        run: () => dataProvider.getAllRecommendations(payloadBase),
      }
    case 'discovery':
      return {
        source: 'discovery',
        run: () => dataProvider.getDiscoveryRecommendations(payloadBase),
      }
    case 'text': {
      const text = autoplay?.textPrompt?.trim()
      if (!text) {
        return null
      }
      return {
        source: 'text',
        run: () =>
          dataProvider.getTextRecommendations({
            ...payloadBase,
            text,
          }),
      }
    }
    default:
      return null
  }
}

const isUsableRecommendationResponse = (data) =>
  data?.resultSource === 'semantic' && data?.degraded !== true

const toTrackMap = (tracks) => {
  const trackMap = {}
  const ids = []
  tracks.forEach((track) => {
    if (!track?.id || trackMap[track.id]) {
      return
    }
    trackMap[track.id] = track
    ids.push(track.id)
  })
  return { trackMap, ids }
}

const filterFreshTracks = (tracks, player, autoplay) => {
  const seen = new Set([
    ...getAllQueueTrackIds(player),
    ...(autoplay?.playedTrackIds || []),
    ...(autoplay?.requestedTrackIds || []),
    ...(autoplay?.negativeTrackIds || []),
  ])

  return (tracks || []).filter((track) => {
    const trackId = track?.id?.toString?.().trim?.() || ''
    if (!trackId || seen.has(trackId)) {
      return false
    }
    seen.add(trackId)
    return true
  })
}

const appendRepeatQueue = (dispatch, player, source) => {
  const repeatedSongs = getNonRadioQueueItems(player?.queue || [])
    .map((item) => item.song)
    .filter(Boolean)

  if (repeatedSongs.length === 0) {
    return false
  }

  dispatch(
    markAutoPlayTracksRequested(
      repeatedSongs.map((song) => song.mediaFileId || song.id).filter(Boolean),
      `${source}:repeat`,
    ),
  )
  dispatch(appendSongs(repeatedSongs))
  return true
}

const notifyWarnings = (warnings, notify, translate) => {
  if (typeof notify !== 'function' || !Array.isArray(warnings)) {
    return
  }
  warnings.forEach((warning) => {
    notify(sanitizeRecommendationWarning(warning, translate), { type: 'info' })
  })
}

export const refillAutoPlayQueue = async ({
  autoplay,
  dataProvider,
  dispatch,
  notify,
  player,
  silent = true,
  source = 'auto',
  translate = (value, options = {}) => options._ || value,
}) => {
  if (!dataProvider || typeof dispatch !== 'function') {
    return { status: 'unavailable' }
  }
  if (autoplay?.fetching) {
    return { status: 'busy' }
  }

  dispatch(startAutoPlayFetch())

  try {
    const payloadBase = buildPayloadBase(player, autoplay)
    const attempts = []
    const seedIds = buildQueueSeedTrackIds(player, autoplay)
    if (seedIds.length > 0) {
      attempts.push({
        requestSource: 'custom',
        run: () =>
          dataProvider.getCustomRecommendations({
            ...payloadBase,
            songIds: seedIds,
          }),
      })
    }
    const fallback = buildFallbackRequest(dataProvider, autoplay, payloadBase)
    if (fallback) {
      attempts.push({
        requestSource: fallback.source,
        run: fallback.run,
      })
    }

    for (const attempt of attempts) {
      try {
        const { data } = await attempt.run()
        if (!isUsableRecommendationResponse(data)) {
          continue
        }
        const freshTracks = filterFreshTracks(
          data?.tracks || [],
          player,
          autoplay,
        )
        if (freshTracks.length === 0) {
          continue
        }
        const { trackMap, ids } = toTrackMap(freshTracks)
        if (ids.length === 0) {
          continue
        }
        dispatch(markAutoPlayTracksRequested(ids, attempt.requestSource))
        if ((player?.queue || []).length === 0) {
          dispatch(playTracks(trackMap, ids, ids[0]))
        } else {
          dispatch(addTracks(trackMap, ids))
        }
        if (!silent) {
          notifyWarnings(data?.warnings, notify, translate)
        }
        return {
          status: 'recommended',
          source: attempt.requestSource,
          trackIds: ids,
        }
      } catch (error) {
        if (!silent && typeof notify === 'function') {
          notify(
            formatRecommendationError(
              error,
              translate,
              translate('ra.page.error', {
                _: 'Unable to load the next songs.',
              }),
            ),
            { type: 'warning' },
          )
        }
      }
    }

    if (appendRepeatQueue(dispatch, player, source)) {
      return { status: 'repeated', source: `${source}:repeat` }
    }

    return { status: 'empty' }
  } finally {
    dispatch(finishAutoPlayFetch())
  }
}
