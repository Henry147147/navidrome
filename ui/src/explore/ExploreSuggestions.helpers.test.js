import { describe, it, expect } from 'vitest'
import {
  isTextModel,
  uniqueStrings,
  clampMinAgreement,
  buildTextRequestModels,
} from './recommendationModelUtils'

describe('ExploreSuggestions helper functions', () => {
  it('detects text-capable models', () => {
    expect(isTextModel('muq_mulan')).toBe(true)
    expect(isTextModel('muq_audio')).toBe(false)
  })

  it('deduplicates non-empty strings while preserving order', () => {
    const result = uniqueStrings([
      'muq_audio',
      '',
      'muq_audio',
      'muq_mulan',
      null,
    ])
    expect(result).toEqual(['muq_audio', 'muq_mulan'])
  })

  it('clamps minimum agreement to valid range', () => {
    expect(clampMinAgreement(-1, 3)).toBe(1)
    expect(clampMinAgreement('bad', 3)).toBe(1)
    expect(clampMinAgreement(2, 3)).toBe(2)
    expect(clampMinAgreement(9, 2)).toBe(2)
  })

  it('builds hybrid text+song request model list', () => {
    const models = buildTextRequestModels(['muq_audio'], true)
    expect(models).toEqual(['muq_audio', 'muq_mulan'])
  })

  it('builds text-only model list without audio-only models when no song seeds', () => {
    const models = buildTextRequestModels(['muq_audio', 'muq_mulan'], false)
    expect(models).toEqual(['muq_mulan'])
  })

  it('falls back to muq_mulan for text-only requests when only audio is selected', () => {
    const models = buildTextRequestModels(['muq_audio'], false)
    expect(models).toEqual(['muq_mulan'])
  })

  it('falls back to MuQ defaults when inputs are empty', () => {
    expect(buildTextRequestModels([], false)).toEqual(['muq_mulan'])
    expect(buildTextRequestModels([], true)).toEqual(['muq_audio', 'muq_mulan'])
  })
})
