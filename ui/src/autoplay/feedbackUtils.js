const normalizeTrackId = (trackId) => {
  if (trackId === null || trackId === undefined) {
    return ''
  }
  return trackId.toString().trim()
}

const toUniqueSet = (values = []) => {
  const result = new Set()
  values.forEach((value) => {
    const normalized = normalizeTrackId(value)
    if (normalized) {
      result.add(normalized)
    }
  })
  return result
}

export const computeFeedback = (
  positives = [],
  negatives = [],
  trackId,
  direction,
) => {
  const positiveSet = toUniqueSet(positives)
  const negativeSet = toUniqueSet(negatives)
  const normalized = normalizeTrackId(trackId)

  if (!normalized) {
    return {
      positive: Array.from(positiveSet),
      negative: Array.from(negativeSet),
    }
  }

  if (direction === 'up') {
    if (positiveSet.has(normalized)) {
      positiveSet.delete(normalized)
    } else {
      positiveSet.add(normalized)
    }
    negativeSet.delete(normalized)
  }

  if (direction === 'down') {
    if (negativeSet.has(normalized)) {
      negativeSet.delete(normalized)
    } else {
      negativeSet.add(normalized)
    }
    positiveSet.delete(normalized)
  }

  return {
    positive: Array.from(positiveSet),
    negative: Array.from(negativeSet),
  }
}

export const ensureUniqueFeedback = (positives = [], negatives = []) => {
  return {
    positive: Array.from(toUniqueSet(positives)),
    negative: Array.from(toUniqueSet(negatives)),
  }
}

export default computeFeedback
