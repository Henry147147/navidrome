import { describe, expect, it, vi } from 'vitest'
import {
  buildOverrideKey,
  decorateQueueWithOverride,
  toStreamQuery,
} from './streamingOverrideUtils'

describe('streamingOverrideUtils', () => {
  it('returns null query for default mode', () => {
    expect(
      toStreamQuery({
        mode: 'default',
        forceTranscoding: false,
        profileId: null,
        format: null,
        maxBitRate: null,
      }),
    ).toBeNull()
  })

  it('returns stream query for override mode', () => {
    expect(
      toStreamQuery({
        mode: 'override',
        forceTranscoding: true,
        profileId: 'tr_opus',
        format: 'opus',
        maxBitRate: 192,
      }),
    ).toEqual({ format: 'opus', maxBitRate: 192 })
  })

  it('returns null query when force toggle is disabled', () => {
    expect(
      toStreamQuery({
        mode: 'override',
        forceTranscoding: false,
        profileId: 'tr_opus',
        format: 'opus',
        maxBitRate: 192,
      }),
    ).toBeNull()
  })

  it('builds stable override key', () => {
    expect(
      buildOverrideKey({
        mode: 'override',
        forceTranscoding: true,
        profileId: 'tr_opus',
        format: 'opus',
        maxBitRate: 192,
      }),
    ).toBe('opus:192')
    expect(buildOverrideKey({ mode: 'default' })).toBe('default')
    expect(
      buildOverrideKey({
        mode: 'override',
        forceTranscoding: false,
        format: 'opus',
        maxBitRate: 192,
      }),
    ).toBe('default')
  })

  it('leaves radio entries unchanged', () => {
    const queue = [{ uuid: 'r1', isRadio: true, streamUrl: '/radio' }]
    const builder = vi.fn()
    const result = decorateQueueWithOverride(
      queue,
      'opus:192',
      { format: 'opus', maxBitRate: 192 },
      builder,
    )

    expect(result[0]).toBe(queue[0])
    expect(builder).not.toHaveBeenCalled()
  })

  it('regenerates stream URL when override key changes', () => {
    const queue = [
      {
        uuid: 'q1',
        trackId: 'song-1',
        musicSrc: '/rest/stream?id=song-1',
      },
    ]
    const builder = vi.fn((id, options) => {
      if (!options) {
        return `/rest/stream?id=${id}`
      }
      return `/rest/stream?id=${id}&format=${options.format}&maxBitRate=${options.maxBitRate}`
    })

    const result = decorateQueueWithOverride(
      queue,
      'opus:192',
      { format: 'opus', maxBitRate: 192 },
      builder,
    )

    expect(builder).toHaveBeenCalledWith('song-1', {
      format: 'opus',
      maxBitRate: 192,
    })
    expect(result[0].musicSrc).toContain('format=opus')
    expect(result[0].streamOverrideKey).toBe('opus:192')
  })

  it('does not regenerate URL when override key is unchanged', () => {
    const queue = [
      {
        uuid: 'q1',
        trackId: 'song-1',
        musicSrc: '/rest/stream?id=song-1&format=opus&maxBitRate=192',
        streamOverrideKey: 'opus:192',
      },
    ]
    const builder = vi.fn()
    const result = decorateQueueWithOverride(
      queue,
      'opus:192',
      { format: 'opus', maxBitRate: 192 },
      builder,
    )

    expect(result[0]).toBe(queue[0])
    expect(builder).not.toHaveBeenCalled()
  })

  it('falls back to existing default URL when queue has no override key', () => {
    const queue = [
      {
        uuid: 'q1',
        trackId: 'song-1',
        musicSrc: '/rest/stream?id=song-1',
      },
    ]
    const builder = vi.fn()
    const result = decorateQueueWithOverride(queue, 'default', null, builder)

    expect(result[0]).toBe(queue[0])
    expect(builder).not.toHaveBeenCalled()
  })
})
