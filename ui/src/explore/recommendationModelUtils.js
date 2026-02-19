export const isTextModel = (model) =>
  model === 'lyrics' || model === 'description'

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

export const buildTextRequestModels = (
  selectedModels,
  textTargets,
  hasSongSeeds,
) => {
  const baseModels =
    Array.isArray(selectedModels) && selectedModels.length > 0
      ? selectedModels
      : ['flamingo']
  const normalizedTextTargets =
    Array.isArray(textTargets) && textTargets.length > 0
      ? textTargets
      : ['lyrics', 'description']

  if (hasSongSeeds) {
    return uniqueStrings([...baseModels, ...normalizedTextTargets])
  }

  const textOnlyModels = baseModels.filter(isTextModel)
  const merged = uniqueStrings([...textOnlyModels, ...normalizedTextTargets])
  return merged.length > 0 ? merged : ['lyrics', 'description']
}
