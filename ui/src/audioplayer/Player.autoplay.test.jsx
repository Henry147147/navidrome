import React, { useEffect } from 'react'
import { Provider } from 'react-redux'
import { combineReducers, createStore } from 'redux'
import { render, waitFor } from '@testing-library/react'
import { autoplayReducer } from '../reducers/autoplayReducer'
import { settingsReducer } from '../reducers/settingsReducer'
import { Player } from './Player'

const streamUrlMock = vi.fn((id) => `/rest/stream?id=${id}`)
const mockDataProvider = {
  getAutoPlaySettings: vi.fn().mockResolvedValue({
    data: {
      enabled: true,
      mode: 'recent',
      textPrompt: '',
      excludePlaylistIds: [],
      batchSize: 5,
      diversityOverride: null,
    },
  }),
  getCustomRecommendations: vi.fn().mockResolvedValue({
    data: {
      resultSource: 'semantic',
      degraded: false,
      tracks: [
        { id: 'fresh-1', title: 'Fresh Track 1' },
        { id: 'fresh-2', title: 'Fresh Track 2' },
        { id: 'fresh-3', title: 'Fresh Track 3' },
        { id: 'fresh-4', title: 'Fresh Track 4' },
      ],
    },
  }),
  getOne: vi.fn(() => Promise.resolve({})),
}

let mockAudioInstance

vi.mock('react-admin', () => ({
  createMuiTheme: (theme) => theme || {},
  useAuthState: () => ({ authenticated: true }),
  useDataProvider: () => mockDataProvider,
  useTranslate: () => (key) => key,
}))

vi.mock('react-ga', () => ({
  default: { event: vi.fn() },
}))

vi.mock('react-hotkeys', () => ({
  GlobalHotKeys: () => <div data-testid="global-hotkeys" />,
}))

vi.mock('navidrome-music-player/assets/index.css', () => ({}))

vi.mock('navidrome-music-player', () => ({
  default: function MockMusicPlayer(props) {
    useEffect(() => {
      props.getAudioInstance?.(mockAudioInstance)
    }, [props])
    return <div data-testid="mock-music-player" />
  },
}))

vi.mock('../themes/useCurrentTheme', () => ({
  default: () => ({ player: { theme: 'dark' } }),
}))

vi.mock('./styles', () => ({
  default: () => ({ player: 'player-class' }),
}))

vi.mock('./locale', () => ({
  default: () => ({}),
}))

vi.mock('./keyHandlers', () => ({
  default: () => ({}),
}))

vi.mock('../utils', () => ({
  sendNotification: vi.fn(),
}))

vi.mock('../utils/calculateReplayGain', () => ({
  calculateGain: () => 1,
}))

vi.mock('../subsonic', () => ({
  default: {
    streamUrl: (...args) => streamUrlMock(...args),
    scrobble: vi.fn(),
    nowPlaying: vi.fn(),
  },
}))

describe('<Player /> autoplay refill', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockAudioInstance = {
      currentTime: 0,
      paused: false,
      pause: vi.fn(),
      volume: 1,
    }
  })

  it('requests queue-seeded recommendations from the always-mounted player', async () => {
    const playerReducer = (
      state = {
        queue: [
          {
            uuid: 'uuid-1',
            trackId: 'song-1',
            musicSrc: '/rest/stream?id=song-1',
            song: { id: 'song-1', title: 'Song 1', artist: 'Artist 1' },
          },
          {
            uuid: 'uuid-2',
            trackId: 'song-2',
            musicSrc: '/rest/stream?id=song-2',
            song: { id: 'song-2', title: 'Song 2', artist: 'Artist 2' },
          },
        ],
        current: {
          uuid: 'uuid-2',
          trackId: 'song-2',
          isRadio: false,
          song: { id: 'song-2', title: 'Song 2', artist: 'Artist 2' },
        },
        clear: false,
        volume: 1,
        mode: 'order',
        playIndex: 1,
        savedPlayIndex: 1,
      },
      action,
    ) => {
      switch (action.type) {
        case 'PLAYER_CURRENT':
          return { ...state, current: action.data || state.current }
        case 'PLAYER_ADD_TRACKS':
          return {
            ...state,
            queue: [
              ...state.queue,
              ...Object.values(action.data).map((track) => ({
                uuid: `uuid-${track.id}`,
                trackId: track.id,
                musicSrc: `/rest/stream?id=${track.id}`,
                song: track,
              })),
            ],
          }
        default:
          return state
      }
    }

    const replayGainReducer = (state = { gainMode: 'off' }) => state
    const store = createStore(
      combineReducers({
        autoplay: autoplayReducer,
        player: playerReducer,
        replayGain: replayGainReducer,
        settings: settingsReducer,
      }),
      {
        autoplay: {
          enabled: true,
          mode: 'recent',
          textPrompt: '',
          excludePlaylistIds: [],
          batchSize: 5,
          diversityOverride: null,
          loaded: true,
          fetching: false,
          playedTrackIds: [],
          requestedTrackIds: [],
          positiveTrackIds: [],
          negativeTrackIds: [],
          lastRefillSource: null,
        },
      },
    )

    render(
      <Provider store={store}>
        <Player />
      </Provider>,
    )

    await waitFor(() => {
      expect(mockDataProvider.getCustomRecommendations).toHaveBeenCalledWith(
        expect.objectContaining({
          songIds: ['song-2', 'song-1'],
        }),
      )
    })
  })
})
