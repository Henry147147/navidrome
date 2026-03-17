import { describe, expect, it, vi } from 'vitest'
import {
  buildExcludeTrackIds,
  buildQueueSeedTrackIds,
  refillAutoPlayQueue,
} from './runtime'

describe('autoplay runtime', () => {
  it('builds queue-context seeds from recent queue history and played tracks', () => {
    const player = {
      queue: [
        { uuid: 'u1', trackId: 'song-1' },
        { uuid: 'u2', trackId: 'song-2' },
        { uuid: 'u3', trackId: 'song-3' },
      ],
      current: { uuid: 'u2' },
    }
    const autoplay = {
      playedTrackIds: ['song-0', 'song-2'],
    }

    expect(buildQueueSeedTrackIds(player, autoplay)).toEqual([
      'song-2',
      'song-1',
      'song-0',
    ])
  })

  it('builds exclusions from played, requested, disliked, and queue tail tracks', () => {
    const player = {
      queue: [
        { uuid: 'u1', trackId: 'song-1' },
        { uuid: 'u2', trackId: 'song-2' },
        { uuid: 'u3', trackId: 'song-3' },
      ],
      current: { uuid: 'u2' },
    }
    const autoplay = {
      playedTrackIds: ['played-1'],
      requestedTrackIds: ['requested-1'],
      negativeTrackIds: ['down-1'],
    }

    expect(buildExcludeTrackIds(player, autoplay)).toEqual([
      'played-1',
      'requested-1',
      'down-1',
      'song-2',
      'song-3',
    ])
  })

  it('requests queue-seeded recommendations and appends only fresh tracks', async () => {
    const dispatch = vi.fn()
    const dataProvider = {
      getCustomRecommendations: vi.fn().mockResolvedValue({
        data: {
          resultSource: 'semantic',
          degraded: false,
          tracks: [
            { id: 'song-3', title: 'Duplicate Tail' },
            { id: 'fresh-1', title: 'Fresh Track' },
          ],
        },
      }),
      getRecentRecommendations: vi.fn(),
    }
    const autoplay = {
      fetching: false,
      batchSize: 5,
      mode: 'recent',
      excludePlaylistIds: ['pl-1'],
      positiveTrackIds: ['liked-1'],
      negativeTrackIds: ['down-1'],
      playedTrackIds: ['played-1'],
      requestedTrackIds: ['requested-1'],
    }
    const player = {
      queue: [
        { uuid: 'u1', trackId: 'song-1', song: { id: 'song-1' } },
        { uuid: 'u2', trackId: 'song-2', song: { id: 'song-2' } },
        { uuid: 'u3', trackId: 'song-3', song: { id: 'song-3' } },
      ],
      current: { uuid: 'u2' },
    }

    await refillAutoPlayQueue({
      autoplay,
      dataProvider,
      dispatch,
      player,
      source: 'player',
    })

    expect(dataProvider.getCustomRecommendations).toHaveBeenCalledWith(
      expect.objectContaining({
        songIds: ['song-2', 'song-1', 'played-1'],
        excludeTrackIds: expect.arrayContaining([
          'played-1',
          'requested-1',
          'down-1',
          'song-2',
          'song-3',
        ]),
        excludePlaylistIds: ['pl-1'],
        positiveTrackIds: ['liked-1'],
        negativeTrackIds: ['down-1'],
      }),
    )
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'PLAYER_ADD_TRACKS',
        data: { 'fresh-1': { id: 'fresh-1', title: 'Fresh Track' } },
      }),
    )
  })

  it('falls back to repeating the queue when recommendations cannot extend it', async () => {
    const dispatch = vi.fn()
    const dataProvider = {
      getCustomRecommendations: vi.fn().mockResolvedValue({
        data: {
          resultSource: 'semantic',
          degraded: false,
          tracks: [],
        },
      }),
      getRecentRecommendations: vi.fn().mockResolvedValue({
        data: {
          resultSource: 'semantic',
          degraded: false,
          tracks: [],
        },
      }),
    }
    const autoplay = {
      fetching: false,
      batchSize: 5,
      mode: 'recent',
      excludePlaylistIds: [],
      positiveTrackIds: [],
      negativeTrackIds: [],
      playedTrackIds: [],
      requestedTrackIds: [],
    }
    const player = {
      queue: [
        {
          uuid: 'u1',
          trackId: 'song-1',
          song: { id: 'song-1', title: 'Song 1' },
        },
        {
          uuid: 'u2',
          trackId: 'song-2',
          song: { id: 'song-2', title: 'Song 2' },
        },
      ],
      current: { uuid: 'u2' },
    }

    const result = await refillAutoPlayQueue({
      autoplay,
      dataProvider,
      dispatch,
      player,
      source: 'page',
    })

    expect(result).toEqual({
      status: 'repeated',
      source: 'page:repeat',
    })
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'PLAYER_APPEND_SONGS',
        data: [
          { id: 'song-1', title: 'Song 1' },
          { id: 'song-2', title: 'Song 2' },
        ],
      }),
    )
  })
})
