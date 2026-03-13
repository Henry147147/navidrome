import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ExploreSuggestions from './ExploreSuggestions'

const mocked = vi.hoisted(() => ({
  dataProvider: {},
  notify: vi.fn(),
  refresh: vi.fn(),
  translate: (key, options = {}) => options._ || key,
}))

let latestSettingsPanelProps = null

vi.mock('react-admin', async () => {
  const actual = await vi.importActual('react-admin')
  return {
    ...actual,
    Title: () => null,
    useDataProvider: () => mocked.dataProvider,
    useNotify: () => mocked.notify,
    useRefresh: () => mocked.refresh,
    useTranslate: () => mocked.translate,
    useQueryWithStore: () => ({
      data: {},
      loading: false,
      loaded: true,
      error: null,
    }),
  }
})

vi.mock('./TextPlaylistGenerator', () => ({
  default: () => <div data-testid="text-playlist-generator" />,
}))

vi.mock('./ExploreSettingsPanel', () => ({
  default: (props) => {
    latestSettingsPanelProps = props
    return (
      <div data-testid="explore-settings-panel">
        <div data-testid="settings-draft">
          {JSON.stringify(props.draft || {})}
        </div>
        <div data-testid="settings-max-duration-error">
          {props.errors?.maxTrackDurationSeconds || ''}
        </div>
        <button
          type="button"
          onClick={() => {
            props.onFieldChange('minTrackDurationSeconds', 60)
            props.onFieldChange('maxTrackDurationSeconds', 600)
          }}
        >
          set-valid-duration
        </button>
        <button
          type="button"
          onClick={() => {
            props.onFieldChange('minTrackDurationSeconds', 700)
            props.onFieldChange('maxTrackDurationSeconds', 600)
          }}
        >
          set-invalid-duration
        </button>
        <button type="button" onClick={props.onSave}>
          save-settings
        </button>
      </div>
    )
  },
}))

describe('ExploreSuggestions', () => {
  beforeEach(() => {
    latestSettingsPanelProps = null
    mocked.notify.mockReset()
    mocked.refresh.mockReset()
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
      getRecommendationSettings: vi.fn().mockResolvedValue({
        data: {
          mixLength: 30,
          baseDiversity: 0.2,
          discoveryExploration: 0.6,
          seedRecencyWindowDays: 45,
          favoritesBlendWeight: 0.8,
          lowRatingPenalty: 0.85,
          minTrackDurationSeconds: 45,
          maxTrackDurationSeconds: 840,
        },
      }),
      updateRecommendationSettings: vi.fn().mockResolvedValue({
        data: {
          mixLength: 30,
          baseDiversity: 0.2,
          discoveryExploration: 0.6,
          seedRecencyWindowDays: 45,
          favoritesBlendWeight: 0.8,
          lowRatingPenalty: 0.85,
          minTrackDurationSeconds: 60,
          maxTrackDurationSeconds: 600,
        },
      }),
      getRecentRecommendations: vi.fn(),
      getFavoriteRecommendations: vi.fn(),
      getAllRecommendations: vi.fn(),
      getDiscoveryRecommendations: vi.fn(),
      getCustomRecommendations: vi.fn(),
      getTextRecommendations: vi.fn(),
      getList: vi.fn(),
      create: vi.fn(),
    }
  })

  it('loads recommendation settings and passes duration range to settings panel', async () => {
    render(<ExploreSuggestions />)

    await userEvent.click(screen.getByRole('tab', { name: 'Settings' }))

    await waitFor(() => {
      expect(screen.getByTestId('explore-settings-panel')).toBeInTheDocument()
      expect(latestSettingsPanelProps?.draft?.minTrackDurationSeconds).toBe(45)
      expect(latestSettingsPanelProps?.draft?.maxTrackDurationSeconds).toBe(840)
    })
  })

  it('defaults the overview to MuQ models and hides legacy text targets', async () => {
    render(<ExploreSuggestions />)

    await waitFor(() => {
      expect(screen.getByText('muq_audio')).toBeInTheDocument()
      expect(screen.getByText('muq_mulan')).toBeInTheDocument()
    })

    expect(screen.queryByText('Text targets')).not.toBeInTheDocument()
  })

  it('disables semantic generators when recommendation health reports engine unavailable', async () => {
    mocked.dataProvider.getRecommendationHealth.mockResolvedValue({
      data: {
        status: 'unavailable',
        engine: {
          ready: false,
          reasonCode: 'milvus_unreachable',
          message: 'Semantic recommendations are temporarily unavailable.',
        },
        text: {
          ready: false,
          reasonCode: 'text_embedding_unreachable',
          message: 'Text recommendations are currently offline.',
        },
        batch: { ready: false },
        availableModes: [],
        degradedModes: [
          'recent',
          'favorites',
          'all',
          'discovery',
          'custom',
          'text',
        ],
      },
    })

    render(<ExploreSuggestions />)

    await waitFor(() => {
      expect(
        screen.getAllByRole('button', { name: 'Generate mix' })[0],
      ).toBeDisabled()
    })

    expect(screen.getByText('Recommendation system status')).toBeInTheDocument()
  })

  it('saves updated duration settings from settings panel', async () => {
    render(<ExploreSuggestions />)

    await userEvent.click(screen.getByRole('tab', { name: 'Settings' }))
    await waitFor(() =>
      expect(screen.getByTestId('explore-settings-panel')).toBeInTheDocument(),
    )

    await userEvent.click(
      screen.getByRole('button', { name: 'set-valid-duration' }),
    )
    await userEvent.click(screen.getByRole('button', { name: 'save-settings' }))

    await waitFor(() => {
      expect(
        mocked.dataProvider.updateRecommendationSettings,
      ).toHaveBeenCalledTimes(1)
    })

    expect(
      mocked.dataProvider.updateRecommendationSettings,
    ).toHaveBeenCalledWith(
      expect.objectContaining({
        minTrackDurationSeconds: 60,
        maxTrackDurationSeconds: 600,
      }),
    )
  })

  it('blocks save when min duration is greater than max duration', async () => {
    render(<ExploreSuggestions />)

    await userEvent.click(screen.getByRole('tab', { name: 'Settings' }))
    await waitFor(() =>
      expect(screen.getByTestId('explore-settings-panel')).toBeInTheDocument(),
    )

    await userEvent.click(
      screen.getByRole('button', { name: 'set-invalid-duration' }),
    )
    await userEvent.click(screen.getByRole('button', { name: 'save-settings' }))

    expect(
      mocked.dataProvider.updateRecommendationSettings,
    ).not.toHaveBeenCalled()

    await waitFor(() => {
      expect(
        screen.getByTestId('settings-max-duration-error').textContent,
      ).toMatch(/greater than or equal to minimum track length/i)
    })
  })
})
