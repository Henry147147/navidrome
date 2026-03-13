import { useEffect, useState } from 'react'

export const DEFAULT_RECOMMENDATION_HEALTH = {
  status: 'unavailable',
  engine: { ready: false, reasonCode: '', message: '' },
  text: { ready: false, reasonCode: '', message: '' },
  batch: { ready: false, reasonCode: '', message: '' },
  availableModes: [],
  degradedModes: [],
}

export const READY_RECOMMENDATION_HEALTH = {
  status: 'ready',
  engine: { ready: true, reasonCode: '', message: '' },
  text: { ready: true, reasonCode: '', message: '' },
  batch: { ready: true, reasonCode: '', message: '' },
  availableModes: ['recent', 'favorites', 'all', 'discovery', 'custom', 'text'],
  degradedModes: [],
}

export const isRecommendationModeAvailable = (health, mode) =>
  Boolean((health?.availableModes || []).includes(mode))

export const sanitizeRecommendationWarning = (warning, translate) => {
  const normalized = `${warning || ''}`.toLowerCase()
  if (
    normalized.includes('recommendation service unavailable') ||
    normalized.includes('milvus init failed') ||
    normalized.includes('schema mismatch')
  ) {
    return translate('recommendations.warning.engineUnavailable', {
      _: 'Semantic recommendations are temporarily unavailable.',
    })
  }
  if (normalized.includes('could not be matched to tracks')) {
    return translate('recommendations.warning.unresolvedTracks', {
      _: 'Some recommendation candidates could not be matched to tracks in your library.',
    })
  }
  return warning
}

export const formatRecommendationError = (error, translate, fallback) => {
  const code = error?.body?.code
  switch (code) {
    case 'milvus_schema_mismatch':
      return translate('recommendations.error.schemaMismatch', {
        _: 'Semantic recommendations are unavailable until the Milvus embedding dimensions are aligned.',
      })
    case 'milvus_unreachable':
      return translate('recommendations.error.engineUnavailable', {
        _: 'Semantic recommendations are temporarily unavailable.',
      })
    case 'text_embedding_unreachable':
      return translate('recommendations.error.textUnavailable', {
        _: 'Text recommendations are currently offline.',
      })
    case 'batch_service_unreachable':
      return translate('recommendations.error.batchUnavailable', {
        _: 'Batch embedding service is currently offline.',
      })
    case 'no_semantic_candidates':
      return translate('recommendations.error.noSemanticCandidates', {
        _: 'No semantic matches were found. Try different seeds or fewer exclusions.',
      })
    default:
      return error?.message || fallback
  }
}

export const healthMessageForMode = (health, mode, translate) => {
  if (!health) {
    return translate('recommendations.error.engineUnavailable', {
      _: 'Semantic recommendations are temporarily unavailable.',
    })
  }
  if (mode === 'text') {
    return (
      health.text?.message ||
      translate('recommendations.error.textUnavailable', {
        _: 'Text recommendations are currently offline.',
      })
    )
  }
  return (
    health.engine?.message ||
    translate('recommendations.error.engineUnavailable', {
      _: 'Semantic recommendations are temporarily unavailable.',
    })
  )
}

export const useRecommendationHealth = (dataProvider) => {
  const [health, setHealth] = useState(DEFAULT_RECOMMENDATION_HEALTH)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    if (typeof dataProvider?.getRecommendationHealth !== 'function') {
      setHealth(READY_RECOMMENDATION_HEALTH)
      setLoading(false)
      return () => {
        active = false
      }
    }
    setLoading(true)
    dataProvider
      .getRecommendationHealth()
      .then(({ data }) => {
        if (!active) {
          return
        }
        setHealth({ ...DEFAULT_RECOMMENDATION_HEALTH, ...(data || {}) })
      })
      .catch(() => {
        if (active) {
          setHealth(DEFAULT_RECOMMENDATION_HEALTH)
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [dataProvider])

  return { health, loading }
}
