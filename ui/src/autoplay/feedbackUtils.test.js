import { computeFeedback, ensureUniqueFeedback } from './feedbackUtils'

describe('feedbackUtils', () => {
  it('adds positive feedback and toggles off on repeat', () => {
    const first = computeFeedback([], [], 'track-1', 'up')
    expect(first.positive).toEqual(['track-1'])
    expect(first.negative).toEqual([])

    const second = computeFeedback(
      first.positive,
      first.negative,
      'track-1',
      'up',
    )
    expect(second.positive).toEqual([])
    expect(second.negative).toEqual([])
  })

  it('adds negative feedback and removes conflicting positives', () => {
    const withPositive = computeFeedback([], [], 'track-2', 'up')
    const withNegative = computeFeedback(
      withPositive.positive,
      withPositive.negative,
      'track-2',
      'down',
    )
    expect(withNegative.positive).toEqual([])
    expect(withNegative.negative).toEqual(['track-2'])
  })

  it('normalizes identifiers and ignores empty values', () => {
    const result = computeFeedback(
      ['  track-3  '],
      ['track-4'],
      '  track-4 ',
      'down',
    )
    expect(result.positive).toEqual(['track-3'])
    expect(result.negative).toEqual([])
  })

  it('deduplicates feedback when merging', () => {
    const unique = ensureUniqueFeedback(
      ['track-5', 'track-5', ''],
      ['track-6', 'track-6'],
    )
    expect(unique.positive).toEqual(['track-5'])
    expect(unique.negative).toEqual(['track-6'])
  })
})
