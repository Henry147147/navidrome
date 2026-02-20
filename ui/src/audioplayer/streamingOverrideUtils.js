export const DEFAULT_STREAMING_OVERRIDE = {
  mode: 'default',
  profileId: null,
  format: null,
  maxBitRate: null,
}

const normalizeBitrate = (value) => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null
  }
  return parsed
}

export const buildOverrideKey = (override) => {
  if (override?.mode !== 'override') {
    return 'default'
  }
  const format = override?.format ? String(override.format) : ''
  const maxBitRate = normalizeBitrate(override?.maxBitRate)
  if (!format || maxBitRate === null) {
    return 'default'
  }
  return `${format}:${maxBitRate}`
}

export const toStreamQuery = (override) => {
  if (override?.mode !== 'override') {
    return null
  }
  const format = override?.format ? String(override.format) : ''
  const maxBitRate = normalizeBitrate(override?.maxBitRate)
  if (!format || maxBitRate === null) {
    return null
  }
  return { format, maxBitRate }
}

const resolveTrackId = (item) => {
  if (!item || item.isRadio) {
    return null
  }
  if (item.trackId) {
    return item.trackId
  }
  if (item.song?.mediaFileId) {
    return item.song.mediaFileId
  }
  if (item.song?.id) {
    return item.song.id
  }
  return null
}

export const decorateQueueWithOverride = (
  queue,
  overrideKey,
  streamQuery,
  streamUrlBuilder,
) => {
  if (!Array.isArray(queue)) {
    return []
  }

  return queue.map((item) => {
    if (!item || item.isRadio) {
      return item
    }

    const currentKey = item.streamOverrideKey
    if (overrideKey === 'default') {
      if (currentKey === undefined || currentKey === 'default') {
        return item
      }
    } else if (currentKey === overrideKey) {
      return item
    }

    const trackId = resolveTrackId(item)
    if (!trackId) {
      return item
    }

    const nextSrc =
      overrideKey === 'default'
        ? streamUrlBuilder(trackId)
        : streamUrlBuilder(trackId, streamQuery)

    return {
      ...item,
      musicSrc: nextSrc,
      streamOverrideKey: overrideKey,
    }
  })
}
