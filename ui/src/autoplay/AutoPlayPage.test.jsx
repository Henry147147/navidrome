import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import AutoPlayPage from './AutoPlayPage'

const mocked = vi.hoisted(() => ({
  dataProvider: {},
  notify: vi.fn(),
  dispatch: vi.fn(),
  translate: (key, options = {}) => options._ || key,
  storeState: {
    player: {
      queue: Array.from({ length: 5 }, (_, index) => ({
        uuid: `uuid-${index + 1}`,
        trackId: `existing-${index + 1}`,
        song: {
          id: `existing-${index + 1}`,
          title: `Existing ${index + 1}`,
          artist: 'Existing Artist',
          album: 'Existing Album',
        },
      })),
      current: {
        uuid: 'uuid-1',
        trackId: 'existing-1',
        song: {
          id: 'existing-1',
          title: 'Existing 1',
          artist: 'Existing Artist',
          album: 'Existing Album',
        },
      },
    },
  },
}))

vi.mock('react-admin', async () => {
  const actual = await vi.importActual('react-admin')
  return {
    ...actual,
    Title: () => null,
    useDataProvider: () => mocked.dataProvider,
    useNotify: () => mocked.notify,
    useTranslate: () => mocked.translate,
  }
})

vi.mock('react-redux', () => ({
  useDispatch: () => mocked.dispatch,
  useSelector: (selector) => selector(mocked.storeState),
}))

vi.mock('../actions', () => ({
  addTracks: (trackMap, ids) => ({
    type: 'ADD_TRACKS',
    payload: { trackMap, ids },
  }),
  playTracks: (trackMap, ids, currentId) => ({
    type: 'PLAY_TRACKS',
    payload: { trackMap, ids, currentId },
  }),
  clearQueue: () => ({ type: 'CLEAR_QUEUE' }),
}))

describe('AutoPlayPage', () => {
  beforeEach(() => {
    mocked.notify.mockReset()
    mocked.dispatch.mockReset()
    mocked.storeState = {
      player: {
        queue: Array.from({ length: 5 }, (_, index) => ({
          uuid: `uuid-${index + 1}`,
          trackId: `existing-${index + 1}`,
          song: {
            id: `existing-${index + 1}`,
            title: `Existing ${index + 1}`,
            artist: 'Existing Artist',
            album: 'Existing Album',
          },
        })),
        current: {
          uuid: 'uuid-1',
          trackId: 'existing-1',
          song: {
            id: 'existing-1',
            title: 'Existing 1',
            artist: 'Existing Artist',
            album: 'Existing Album',
          },
        },
      },
    }
    mocked.dataProvider = {
      getRecommendationHealth: vi.fn().mockResolvedValue({
        data: {
          status: 'ready',
          engine: { ready: true },
          text: { ready: true },
          batch: { ready: true },
          availableModes: [
            'recent',
            'favorites',
            'all',
            'discovery',
            'custom',
            'text',
          ],
          degradedModes: [],
        },
      }),
      getAutoPlaySettings: vi.fn().mockResolvedValue({
        data: {
          mode: 'recent',
          textPrompt: '',
          excludePlaylistIds: [],
          diversityOverride: null,
        },
      }),
      getList: vi.fn().mockResolvedValue({ data: [], total: 0 }),
      getRecentRecommendations: vi.fn(),
      getFavoriteRecommendations: vi.fn(),
      getAllRecommendations: vi.fn(),
      getDiscoveryRecommendations: vi.fn(),
      getCustomRecommendations: vi.fn(),
      getTextRecommendations: vi.fn(),
      updateAutoPlaySettings: vi.fn(),
    }
  })

  it('excludes already requested tracks on subsequent fetches', async () => {
    mocked.dataProvider.getRecentRecommendations
      .mockResolvedValueOnce({
        data: {
          resultSource: 'semantic',
          degraded: false,
          tracks: [
            {
              id: 'track-a',
              title: 'Track A',
              artist: 'Artist A',
              album: 'Album A',
            },
          ],
          warnings: [],
        },
      })
      .mockResolvedValueOnce({
        data: {
          resultSource: 'semantic',
          degraded: false,
          tracks: [
            {
              id: 'track-b',
              title: 'Track B',
              artist: 'Artist B',
              album: 'Album B',
            },
          ],
          warnings: [],
        },
      })

    render(<AutoPlayPage />)

    await waitFor(() => {
      expect(mocked.dataProvider.getAutoPlaySettings).toHaveBeenCalledTimes(1)
    })

    await userEvent.click(
      screen.getByRole('button', { name: 'Start Auto Play' }),
    )

    await waitFor(() => {
      expect(
        mocked.dataProvider.getRecentRecommendations,
      ).toHaveBeenCalledTimes(1)
    })

    await userEvent.click(screen.getByRole('button', { name: 'Add more' }))

    await waitFor(() => {
      expect(
        mocked.dataProvider.getRecentRecommendations,
      ).toHaveBeenCalledTimes(2)
    })

    const secondPayload =
      mocked.dataProvider.getRecentRecommendations.mock.calls[1][0]
    expect(secondPayload.excludeTrackIds).toContain('track-a')
    expect(mocked.notify).not.toHaveBeenCalledWith(
      'pages.autoplay.notifications.noNew',
      expect.any(Object),
    )
  })

  it('still notifies when no unseen tracks are available', async () => {
    mocked.dataProvider.getRecentRecommendations.mockResolvedValue({
      data: {
        resultSource: 'semantic',
        degraded: false,
        tracks: [],
        warnings: [],
      },
    })

    render(<AutoPlayPage />)

    await waitFor(() => {
      expect(mocked.dataProvider.getAutoPlaySettings).toHaveBeenCalledTimes(1)
    })

    await userEvent.click(
      screen.getByRole('button', { name: 'Start Auto Play' }),
    )

    await waitFor(() => {
      expect(mocked.notify).toHaveBeenCalledWith(
        'pages.autoplay.notifications.noNew',
        { type: 'warning' },
      )
    })
  })

  it('uses semantic text recommendations instead of song search for text mode', async () => {
    mocked.dataProvider.getAutoPlaySettings.mockResolvedValue({
      data: {
        mode: 'text',
        textPrompt: 'late night jazz',
        excludePlaylistIds: [],
        diversityOverride: null,
      },
    })
    mocked.dataProvider.getTextRecommendations.mockResolvedValue({
      data: {
        resultSource: 'semantic',
        degraded: false,
        tracks: [
          {
            id: 'track-text-1',
            title: 'Track Text 1',
            artist: 'Artist Text 1',
            album: 'Album Text 1',
          },
        ],
        warnings: [],
      },
    })

    render(<AutoPlayPage />)

    await waitFor(() => {
      expect(mocked.dataProvider.getAutoPlaySettings).toHaveBeenCalledTimes(1)
    })

    await userEvent.click(
      screen.getByRole('button', { name: 'Start Auto Play' }),
    )

    await waitFor(() => {
      expect(mocked.dataProvider.getTextRecommendations).toHaveBeenCalledTimes(
        1,
      )
    })

    expect(mocked.dataProvider.getList).not.toHaveBeenCalledWith(
      'song',
      expect.objectContaining({
        filter: expect.objectContaining({ q: 'late night jazz' }),
      }),
    )
  })
})
