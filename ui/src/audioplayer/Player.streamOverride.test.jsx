import React, { useEffect } from 'react'
import { Provider } from 'react-redux'
import { act } from 'react-dom/test-utils'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { combineReducers, createStore } from 'redux'
import { setStreamingOverride } from '../actions'
import { settingsReducer } from '../reducers/settingsReducer'
import { Player } from './Player'

const streamUrlMock = vi.fn((id, options) => {
  if (!options) {
    return `/rest/stream?id=${id}`
  }
  return `/rest/stream?id=${id}&format=${options.format}&maxBitRate=${options.maxBitRate}`
})
const scrobbleMock = vi.fn()
const nowPlayingMock = vi.fn()

let capturedPlayerProps
let mockAudioInstance

vi.mock('react-admin', () => ({
  createMuiTheme: (theme) => theme || {},
  useAuthState: () => ({ authenticated: true }),
  useDataProvider: () => ({ getOne: vi.fn(() => Promise.resolve({})) }),
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
  default: (props) => {
    capturedPlayerProps = props
    useEffect(() => {
      props.getAudioInstance?.(mockAudioInstance)
    }, [props])
    return (
      <button
        data-testid="mock-music-player"
        onClick={() =>
          props.onAudioPlay?.({
            trackId: 'song-1',
            isRadio: false,
            duration: 180,
            currentTime: 0,
            song: { title: 'Song 1', artist: 'Artist 1', album: 'Album 1' },
            cover: '',
            volume: 1,
          })
        }
      >
        play
      </button>
    )
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
    scrobble: (...args) => scrobbleMock(...args),
    nowPlaying: (...args) => nowPlayingMock(...args),
  },
}))

const createTestStore = () => {
  const initialPlayerState = {
    queue: [
      {
        uuid: 'uuid-1',
        trackId: 'song-1',
        musicSrc: '/rest/stream?id=song-1',
        name: 'Song 1',
        song: { title: 'Song 1', artist: 'Artist 1', album: 'Album 1' },
      },
    ],
    current: {
      uuid: 'uuid-1',
      trackId: 'song-1',
      isRadio: false,
      song: { title: 'Song 1', artist: 'Artist 1', album: 'Album 1' },
    },
    clear: false,
    volume: 1,
    mode: 'order',
    playIndex: 0,
    savedPlayIndex: 0,
  }

  const playerReducer = (state = initialPlayerState, action) => {
    if (action.type === 'PLAYER_CURRENT') {
      return {
        ...state,
        current: action.data || state.current,
      }
    }
    return state
  }

  const replayGainReducer = (state = { gainMode: 'off' }) => state

  return createStore(
    combineReducers({
      player: playerReducer,
      settings: settingsReducer,
      replayGain: replayGainReducer,
    }),
  )
}

describe('<Player /> streaming override', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    capturedPlayerProps = null
    mockAudioInstance = {
      currentTime: 0,
      paused: false,
      pause: vi.fn(),
      volume: 1,
    }
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('applies and removes stream query params when override changes', async () => {
    const store = createTestStore()

    render(
      <Provider store={store}>
        <Player />
      </Provider>,
    )

    await waitFor(() => {
      expect(capturedPlayerProps).toBeTruthy()
      expect(capturedPlayerProps.audioLists[0].musicSrc).toBe(
        '/rest/stream?id=song-1',
      )
    })

    act(() => {
      store.dispatch(
        setStreamingOverride({
          mode: 'override',
          profileId: 'tr_opus',
          format: 'opus',
          maxBitRate: 192,
        }),
      )
    })

    await waitFor(() => {
      expect(capturedPlayerProps.audioLists[0].musicSrc).toContain(
        'format=opus',
      )
      expect(capturedPlayerProps.audioLists[0].musicSrc).toContain(
        'maxBitRate=192',
      )
    })

    act(() => {
      store.dispatch(
        setStreamingOverride({
          mode: 'default',
          profileId: null,
          format: null,
          maxBitRate: null,
        }),
      )
    })

    await waitFor(() => {
      expect(capturedPlayerProps.audioLists[0].musicSrc).toBe(
        '/rest/stream?id=song-1',
      )
    })
  })

  it('restores playback position and paused state on immediate override switch', async () => {
    vi.useFakeTimers()
    const store = createTestStore()
    mockAudioInstance.currentTime = 87
    mockAudioInstance.paused = true

    render(
      <Provider store={store}>
        <Player />
      </Provider>,
    )

    act(() => {
      store.dispatch(
        setStreamingOverride({
          mode: 'override',
          profileId: 'tr_opus',
          format: 'opus',
          maxBitRate: 160,
        }),
      )
    })

    await waitFor(() => {
      expect(capturedPlayerProps.audioLists[0].musicSrc).toContain(
        'format=opus',
      )
    })

    fireEvent.click(screen.getByTestId('mock-music-player'))
    expect(mockAudioInstance.currentTime).toBe(87)

    act(() => {
      vi.runAllTimers()
    })
    expect(mockAudioInstance.pause).toHaveBeenCalled()
  })
})
