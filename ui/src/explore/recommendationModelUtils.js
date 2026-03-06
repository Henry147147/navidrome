const DEFAULT_HYBRID_MODELS = ['muq_audio', 'muq_mulan']
const DEFAULT_TEXT_MODELS = ['muq_mulan']

export const isTextModel = (model) => model === 'muq_mulan'

export const uniqueStrings = (values) => {
  const seen = new Set()
  return (values || []).filter((value) => {
    if (!value || seen.has(value)) {
      return false
    }
    seen.add(value)
    return true
  })
}

export const clampMinAgreement = (value, modelCount) => {
  const count = Math.max(modelCount, 1)
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 1) {
    return 1
  }
  if (parsed > count) {
    return count
  }
  return parsed
}

export const buildTextRequestModels = (selectedModels, hasSongSeeds) => {
  const baseModels =
    Array.isArray(selectedModels) && selectedModels.length > 0
      ? selectedModels
      : DEFAULT_HYBRID_MODELS

  if (hasSongSeeds) {
    return uniqueStrings([...baseModels, ...DEFAULT_TEXT_MODELS])
  }

  const textOnlyModels = baseModels.filter(isTextModel)
  const merged = uniqueStrings(textOnlyModels)
  return merged.length > 0 ? merged : DEFAULT_TEXT_MODELS
}
