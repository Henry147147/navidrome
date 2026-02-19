import { describe, it, expect } from 'vitest'
import {
  isTextModel,
  uniqueStrings,
  clampMinAgreement,
  buildTextRequestModels,
} from './recommendationModelUtils'

describe('ExploreSuggestions helper functions', () => {
  it('detects text-capable models', () => {
    expect(isTextModel('lyrics')).toBe(true)
    expect(isTextModel('description')).toBe(true)
    expect(isTextModel('flamingo')).toBe(false)
  })

  it('deduplicates non-empty strings while preserving order', () => {
    const result = uniqueStrings(['lyrics', '', 'lyrics', 'description', null])
    expect(result).toEqual(['lyrics', 'description'])
  })

  it('clamps minimum agreement to valid range', () => {
    expect(clampMinAgreement(-1, 3)).toBe(1)
    expect(clampMinAgreement('bad', 3)).toBe(1)
    expect(clampMinAgreement(2, 3)).toBe(2)
    expect(clampMinAgreement(9, 2)).toBe(2)
  })

  it('builds hybrid text+song request model list', () => {
    const models = buildTextRequestModels(
      ['flamingo'],
      ['lyrics', 'description'],
      true,
    )
    expect(models).toEqual(['flamingo', 'lyrics', 'description'])
  })

  it('builds text-only model list without flamingo when no song seeds', () => {
    const models = buildTextRequestModels(
      ['flamingo'],
      ['lyrics', 'description'],
      false,
    )
    expect(models).toEqual(['lyrics', 'description'])
  })

  it('keeps selected text models for text-only requests', () => {
    const models = buildTextRequestModels(
      ['lyrics'],
      ['description'],
      false,
    )
    expect(models).toEqual(['lyrics', 'description'])
  })

  it('falls back to default text targets when inputs are empty', () => {
    const models = buildTextRequestModels([], [], false)
    expect(models).toEqual(['lyrics', 'description'])
  })
})
