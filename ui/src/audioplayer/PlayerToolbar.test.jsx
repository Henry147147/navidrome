import React from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useMediaQuery } from '@material-ui/core'
import { useGetList, useGetOne, useTranslate } from 'react-admin'
import { useDispatch, useSelector } from 'react-redux'
import { useToggleLove } from '../common'
import { openSaveQueueDialog, setStreamingOverride } from '../actions'
import PlayerToolbar from './PlayerToolbar'

vi.mock('@material-ui/core', async () => {
  const actual = await import('@material-ui/core')
  return {
    ...actual,
    useMediaQuery: vi.fn(),
  }
})

vi.mock('react-admin', () => ({
  useGetOne: vi.fn(),
  useGetList: vi.fn(),
  useTranslate: vi.fn(),
}))

vi.mock('react-redux', () => ({
  useDispatch: vi.fn(),
  useSelector: vi.fn(),
}))

vi.mock('../common', () => ({
  LoveButton: ({ className, disabled }) => (
    <button data-testid="love-button" className={className} disabled={disabled}>
      Love
    </button>
  ),
  useToggleLove: vi.fn(),
}))

vi.mock('../actions', () => ({
  openSaveQueueDialog: vi.fn(),
  setStreamingOverride: vi.fn(),
}))

vi.mock('react-hotkeys', () => ({
  GlobalHotKeys: () => <div data-testid="global-hotkeys" />,
}))

const transcodings = {
  tr_opus: {
    id: 'tr_opus',
    name: 'Opus profile',
    targetFormat: 'opus',
    defaultBitRate: 128,
  },
  tr_mp3: {
    id: 'tr_mp3',
    name: 'MP3 profile',
    targetFormat: 'mp3',
    defaultBitRate: 192,
  },
}

describe('<PlayerToolbar />', () => {
  const mockToggleLove = vi.fn()
  const mockDispatch = vi.fn()
  const mockSongData = { id: 'song-1', name: 'Test Song', starred: false }

  const defaultStoreState = {
    settings: {
      streamingOverride: {
        mode: 'default',
        profileId: null,
        format: null,
        maxBitRate: null,
      },
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    useGetOne.mockReturnValue({ data: mockSongData, loading: false })
    useGetList.mockReturnValue({ data: transcodings, loading: false })
    useToggleLove.mockReturnValue([mockToggleLove, false])
    useDispatch.mockReturnValue(mockDispatch)
    useSelector.mockImplementation((selector) => selector(defaultStoreState))
    useTranslate.mockReturnValue((key) => key)
    openSaveQueueDialog.mockReturnValue({ type: 'OPEN_SAVE_QUEUE_DIALOG' })
    setStreamingOverride.mockImplementation((payload) => ({
      type: 'SET_STREAMING_OVERRIDE',
      data: payload,
    }))
  })

  afterEach(cleanup)

  describe('Desktop layout', () => {
    beforeEach(() => {
      useMediaQuery.mockReturnValue(true)
    })

    it('renders desktop toolbar with save, stream settings, and love buttons', () => {
      render(<PlayerToolbar id="song-1" />)

      const listItems = screen.getAllByRole('listitem')
      expect(listItems).toHaveLength(1)

      expect(screen.getByTestId('save-queue-button')).toBeInTheDocument()
      expect(screen.getByTestId('stream-settings-button')).toBeInTheDocument()
      expect(screen.getByTestId('love-button')).toBeInTheDocument()
      expect(listItems[0].className).toContain('toolbar')
    })

    it('opens save queue dialog when save button is clicked', () => {
      render(<PlayerToolbar id="song-1" />)

      fireEvent.click(screen.getByTestId('save-queue-button'))

      expect(mockDispatch).toHaveBeenCalledWith({
        type: 'OPEN_SAVE_QUEUE_DIALOG',
      })
    })

    it('opens stream settings and renders default + profile options', () => {
      render(<PlayerToolbar id="song-1" />)

      fireEvent.click(screen.getByTestId('stream-settings-button'))

      const profileSelect = screen.getByTestId('stream-profile-select')
      expect(profileSelect).toBeInTheDocument()
      expect(profileSelect.value).toBe('__default__')
      expect(
        screen.getAllByText('player.streamDefaultBehaviorText').length,
      ).toBeGreaterThan(0)
      expect(screen.getByText('Opus profile')).toBeInTheDocument()
      expect(screen.getByText('MP3 profile')).toBeInTheDocument()
    })

    it('dispatches default override when selecting Default profile', () => {
      useSelector.mockImplementation((selector) =>
        selector({
          settings: {
            streamingOverride: {
              mode: 'override',
              profileId: 'tr_opus',
              format: 'opus',
              maxBitRate: 128,
            },
          },
        }),
      )
      render(<PlayerToolbar id="song-1" />)

      fireEvent.click(screen.getByTestId('stream-settings-button'))
      fireEvent.change(screen.getByTestId('stream-profile-select'), {
        target: { value: '__default__' },
      })

      expect(mockDispatch).toHaveBeenCalledWith({
        type: 'SET_STREAMING_OVERRIDE',
        data: {
          mode: 'default',
          profileId: null,
          format: null,
          maxBitRate: null,
        },
      })
    })

    it('dispatches override payload when selecting a profile', () => {
      render(<PlayerToolbar id="song-1" />)

      fireEvent.click(screen.getByTestId('stream-settings-button'))
      fireEvent.change(screen.getByTestId('stream-profile-select'), {
        target: { value: 'tr_opus' },
      })

      expect(mockDispatch).toHaveBeenCalledWith({
        type: 'SET_STREAMING_OVERRIDE',
        data: {
          mode: 'override',
          profileId: 'tr_opus',
          format: 'opus',
          maxBitRate: 128,
        },
      })
    })

    it('dispatches updated bitrate when changing bitrate select', () => {
      useSelector.mockImplementation((selector) =>
        selector({
          settings: {
            streamingOverride: {
              mode: 'override',
              profileId: 'tr_opus',
              format: 'opus',
              maxBitRate: 128,
            },
          },
        }),
      )
      render(<PlayerToolbar id="song-1" />)

      fireEvent.click(screen.getByTestId('stream-settings-button'))
      fireEvent.change(screen.getByTestId('stream-bitrate-select'), {
        target: { value: '192' },
      })

      expect(mockDispatch).toHaveBeenCalledWith({
        type: 'SET_STREAMING_OVERRIDE',
        data: {
          mode: 'override',
          profileId: 'tr_opus',
          format: 'opus',
          maxBitRate: 192,
        },
      })
    })
  })

  describe('Mobile layout', () => {
    beforeEach(() => {
      useMediaQuery.mockReturnValue(false)
    })

    it('renders mobile toolbar with controls in separate list items', () => {
      render(<PlayerToolbar id="song-1" />)

      const listItems = screen.getAllByRole('listitem')
      expect(listItems).toHaveLength(3)
      expect(screen.getByTestId('save-queue-button')).toBeInTheDocument()
      expect(screen.getByTestId('stream-settings-button')).toBeInTheDocument()
      expect(screen.getByTestId('love-button')).toBeInTheDocument()
      expect(listItems[0].className).toContain('mobileListItem')
      expect(listItems[1].className).toContain('mobileListItem')
      expect(listItems[2].className).toContain('mobileListItem')
    })
  })

  describe('Common behavior', () => {
    it('gracefully handles empty transcoding list', () => {
      useGetList.mockReturnValue({ data: {}, loading: false })
      render(<PlayerToolbar id="song-1" />)

      fireEvent.click(screen.getByTestId('stream-settings-button'))
      const profileSelect = screen.getByTestId('stream-profile-select')
      expect(profileSelect).toBeInTheDocument()
      expect(profileSelect.querySelectorAll('option')).toHaveLength(1)
    })

    it('disables controls when no track id is provided', () => {
      render(<PlayerToolbar />)
      expect(screen.getByTestId('love-button')).toBeDisabled()
      expect(screen.getByTestId('stream-settings-button')).toBeDisabled()
    })

    it('renders global hotkeys in both layouts', () => {
      useMediaQuery.mockReturnValue(true)
      render(<PlayerToolbar id="song-1" />)
      expect(screen.getByTestId('global-hotkeys')).toBeInTheDocument()
      cleanup()

      useMediaQuery.mockReturnValue(false)
      render(<PlayerToolbar id="song-1" />)
      expect(screen.getByTestId('global-hotkeys')).toBeInTheDocument()
    })
  })
})
