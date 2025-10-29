import React from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import AutoPlaySettingsPanel from './AutoPlaySettingsPanel'

const translate = (key, options = {}) => options._ || key

describe('AutoPlaySettingsPanel', () => {
  const baseDraft = {
    mode: 'recent',
    batchSize: 5,
    diversityOverride: null,
    textPrompt: '',
    excludePlaylistIds: [],
  }

  const modeOptions = [
    { value: 'recent', label: 'Recent listens' },
    { value: 'favorites', label: 'Liked songs' },
  ]

  it('renders without exposing batch size control', () => {
    render(
      <AutoPlaySettingsPanel
        translate={translate}
        draft={baseDraft}
        onFieldChange={vi.fn()}
        onSave={vi.fn()}
        onReset={vi.fn()}
        saving={false}
        loading={false}
        error={null}
        playlists={[]}
        modeOptions={modeOptions}
      />,
    )

    expect(screen.getByLabelText('Default mode')).toBeInTheDocument()
    expect(screen.queryByLabelText(/Batch size/i)).toBeNull()
    expect(screen.getAllByText('Default text prompt').length).toBeGreaterThan(0)
  })

  it('invokes field change when mode selection updates', async () => {
    const onFieldChange = vi.fn()
    render(
      <AutoPlaySettingsPanel
        translate={translate}
        draft={baseDraft}
        onFieldChange={onFieldChange}
        onSave={vi.fn()}
        onReset={vi.fn()}
        saving={false}
        loading={false}
        error={null}
        playlists={[]}
        modeOptions={modeOptions}
      />,
    )

    const select = screen.getByRole('button', { name: 'Default mode' })
    await userEvent.click(select)
    const option = await screen.findByText('Liked songs')
    await userEvent.click(option)
    expect(onFieldChange).toHaveBeenCalledWith('mode', 'favorites')
  })
})
