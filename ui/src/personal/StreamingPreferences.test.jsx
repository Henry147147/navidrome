import React from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useGetList, useTranslate } from 'react-admin'
import { useDispatch, useSelector } from 'react-redux'
import { setStreamingOverride } from '../actions'
import { StreamingPreferences } from './StreamingPreferences'

vi.mock('react-admin', () => ({
  useGetList: vi.fn(),
  useTranslate: vi.fn(),
}))

vi.mock('react-redux', () => ({
  useDispatch: vi.fn(),
  useSelector: vi.fn(),
}))

vi.mock('../actions', () => ({
  setStreamingOverride: vi.fn(),
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

describe('<StreamingPreferences />', () => {
  const mockDispatch = vi.fn()
  const defaultStoreState = {
    settings: {
      streamingOverride: {
        mode: 'default',
        forceTranscoding: false,
        profileId: null,
        format: null,
        maxBitRate: null,
      },
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    useGetList.mockReturnValue({ data: transcodings, loading: false })
    useDispatch.mockReturnValue(mockDispatch)
    useSelector.mockImplementation((selector) => selector(defaultStoreState))
    useTranslate.mockReturnValue((key) => key)
    setStreamingOverride.mockImplementation((payload) => ({
      type: 'SET_STREAMING_OVERRIDE',
      data: payload,
    }))
  })

  afterEach(cleanup)

  it('renders profile, bitrate, and force controls', () => {
    render(<StreamingPreferences />)

    expect(
      screen.getByTestId('personal-stream-profile-select'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('personal-stream-bitrate-select'),
    ).toBeInTheDocument()
    expect(
      screen.getByTestId('personal-stream-force-toggle'),
    ).toBeInTheDocument()
    expect(screen.getByTestId('personal-stream-profile-select').value).toBe(
      '__default__',
    )
    expect(screen.getByTestId('personal-stream-bitrate-select')).toBeDisabled()
    expect(screen.getByTestId('personal-stream-force-toggle')).toBeDisabled()
  })

  it('dispatches override payload when selecting a profile', () => {
    render(<StreamingPreferences />)

    fireEvent.change(screen.getByTestId('personal-stream-profile-select'), {
      target: { value: 'tr_opus' },
    })

    expect(mockDispatch).toHaveBeenCalledWith({
      type: 'SET_STREAMING_OVERRIDE',
      data: {
        mode: 'override',
        forceTranscoding: false,
        profileId: 'tr_opus',
        format: 'opus',
        maxBitRate: 128,
      },
    })
  })

  it('dispatches default payload when selecting default profile', () => {
    useSelector.mockImplementation((selector) =>
      selector({
        settings: {
          streamingOverride: {
            mode: 'override',
            forceTranscoding: true,
            profileId: 'tr_opus',
            format: 'opus',
            maxBitRate: 128,
          },
        },
      }),
    )
    render(<StreamingPreferences />)

    fireEvent.change(screen.getByTestId('personal-stream-profile-select'), {
      target: { value: '__default__' },
    })

    expect(mockDispatch).toHaveBeenCalledWith({
      type: 'SET_STREAMING_OVERRIDE',
      data: {
        mode: 'default',
        forceTranscoding: false,
        profileId: null,
        format: null,
        maxBitRate: null,
      },
    })
  })

  it('dispatches bitrate updates for selected profile', () => {
    useSelector.mockImplementation((selector) =>
      selector({
        settings: {
          streamingOverride: {
            mode: 'override',
            forceTranscoding: true,
            profileId: 'tr_opus',
            format: 'opus',
            maxBitRate: 128,
          },
        },
      }),
    )
    render(<StreamingPreferences />)

    fireEvent.change(screen.getByTestId('personal-stream-bitrate-select'), {
      target: { value: '192' },
    })

    expect(mockDispatch).toHaveBeenCalledWith({
      type: 'SET_STREAMING_OVERRIDE',
      data: {
        mode: 'override',
        forceTranscoding: true,
        profileId: 'tr_opus',
        format: 'opus',
        maxBitRate: 192,
      },
    })
  })

  it('dispatches force toggle changes', () => {
    useSelector.mockImplementation((selector) =>
      selector({
        settings: {
          streamingOverride: {
            mode: 'override',
            forceTranscoding: false,
            profileId: 'tr_opus',
            format: 'opus',
            maxBitRate: 128,
          },
        },
      }),
    )
    render(<StreamingPreferences />)

    fireEvent.click(screen.getByTestId('personal-stream-force-toggle'))

    expect(mockDispatch).toHaveBeenCalledWith({
      type: 'SET_STREAMING_OVERRIDE',
      data: {
        mode: 'override',
        forceTranscoding: true,
        profileId: 'tr_opus',
        format: 'opus',
        maxBitRate: 128,
      },
    })
  })

  it('gracefully handles empty transcoding list', () => {
    useGetList.mockReturnValue({ data: {}, loading: false })
    render(<StreamingPreferences />)

    const profileSelect = screen.getByTestId('personal-stream-profile-select')
    expect(profileSelect.querySelectorAll('option')).toHaveLength(1)
  })
})
