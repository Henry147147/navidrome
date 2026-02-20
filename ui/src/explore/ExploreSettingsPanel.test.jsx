import React from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import ExploreSettingsPanel from './ExploreSettingsPanel'

const translate = (key, options = {}) => options._ || key

describe('ExploreSettingsPanel', () => {
  const baseDraft = {
    mixLength: 25,
    baseDiversity: 0.15,
    discoveryExploration: 0.6,
    seedRecencyWindowDays: 60,
    favoritesBlendWeight: 0.85,
    lowRatingPenalty: 0.85,
    minTrackDurationSeconds: 30,
    maxTrackDurationSeconds: 15 * 60,
  }

  it('renders duration controls and formatted values', () => {
    render(
      <ExploreSettingsPanel
        translate={translate}
        draft={baseDraft}
        errors={{}}
        onFieldChange={vi.fn()}
        onReset={vi.fn()}
        onSave={vi.fn()}
        saving={false}
        dirty={true}
        loading={false}
        serverError={null}
      />,
    )

    expect(screen.getByText('Minimum track length')).toBeInTheDocument()
    expect(screen.getByText('Maximum track length')).toBeInTheDocument()
    expect(screen.getAllByText('30s').length).toBeGreaterThan(0)
    expect(screen.getAllByText('15m').length).toBeGreaterThan(0)
  })

  it('shows validation error text for duration range', () => {
    render(
      <ExploreSettingsPanel
        translate={translate}
        draft={baseDraft}
        errors={{
          maxTrackDurationSeconds:
            'Maximum track length must be greater than or equal to minimum track length.',
        }}
        onFieldChange={vi.fn()}
        onReset={vi.fn()}
        onSave={vi.fn()}
        saving={false}
        dirty={true}
        loading={false}
        serverError={null}
      />,
    )

    expect(
      screen.getByText(
        'Maximum track length must be greater than or equal to minimum track length.',
      ),
    ).toBeInTheDocument()
  })

  it('calls save and reset handlers', async () => {
    const onSave = vi.fn()
    const onReset = vi.fn()
    render(
      <ExploreSettingsPanel
        translate={translate}
        draft={baseDraft}
        errors={{}}
        onFieldChange={vi.fn()}
        onReset={onReset}
        onSave={onSave}
        saving={false}
        dirty={true}
        loading={false}
        serverError={null}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: 'Save settings' }))
    await userEvent.click(screen.getByRole('button', { name: 'Reset' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onReset).toHaveBeenCalledTimes(1)
  })
})
