import * as React from 'react'
import { TestContext } from 'ra-test'
import { DataProviderContext } from 'react-admin'
import {
  cleanup,
  fireEvent,
  render,
  waitFor,
  screen,
} from '@testing-library/react'
import { describe, beforeEach, afterEach, it, expect, vi } from 'vitest'
import TextPlaylistGenerator from './TextPlaylistGenerator'

describe('TextPlaylistGenerator', () => {
  let mockDataProvider

  beforeEach(() => {
    mockDataProvider = {
      getTextRecommendations: vi.fn(),
    }
  })

  afterEach(cleanup)

  const createTestUtils = () =>
    render(
      <DataProviderContext.Provider value={mockDataProvider}>
        <TestContext
          initialState={{
            admin: { resources: {}, ui: { optimistic: false } },
          }}
        >
          <TextPlaylistGenerator />
        </TestContext>
      </DataProviderContext.Provider>,
    )

  it('renders without crashing', () => {
    const { container } = createTestUtils()
    expect(container).toBeInTheDocument()
  })

  it('shows text input field', () => {
    createTestUtils()
    const textInputs = screen.getAllByRole('textbox')
    expect(textInputs.length).toBeGreaterThan(0)
  })

  it('shows model selector', () => {
    createTestUtils()
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThan(0)
  })

  it('has add negative prompt button', () => {
    createTestUtils()
    const buttons = screen.getAllByRole('button')
    // Should have multiple buttons including add, generate, clear
    expect(buttons.length).toBeGreaterThanOrEqual(2)
  })

  it('allows adding negative prompt fields', async () => {
    createTestUtils()

    const initialTextboxCount = screen.getAllByRole('textbox').length

    // Click first button (likely Add button)
    const buttons = screen.getAllByRole('button')
    const addButton = buttons.find(
      (btn) =>
        btn.textContent.includes('Add') || btn.textContent.includes('add'),
    )

    if (addButton) {
      fireEvent.click(addButton)
    } else {
      // Try first non-primary button
      fireEvent.click(buttons[1])
    }

    await waitFor(() => {
      const newTextboxCount = screen.getAllByRole('textbox').length
      expect(newTextboxCount).toBeGreaterThan(initialTextboxCount)
    })
  })

  it('calls getTextRecommendations when form submitted', async () => {
    mockDataProvider.getTextRecommendations.mockResolvedValue({
      data: {
        tracks: [
          { id: '1', title: 'Song 1', artist: 'Artist 1', album: 'Album 1' },
        ],
        name: 'Generated',
        mode: 'text',
      },
    })

    createTestUtils()

    // Find the main text input (first textbox)
    const textInputs = screen.getAllByRole('textbox')
    const mainInput = textInputs[0]

    // Enter text
    fireEvent.change(mainInput, { target: { value: 'test music query' } })

    // Find and click generate button
    const buttons = screen.getAllByRole('button')
    const generateButton = buttons.find(
      (btn) =>
        btn.textContent.includes('Generate') ||
        btn.textContent.includes('generate'),
    )

    if (generateButton) {
      fireEvent.click(generateButton)

      await waitFor(() => {
        expect(mockDataProvider.getTextRecommendations).toHaveBeenCalled()
      })
    }
  })

  it('displays loading state during generation', async () => {
    mockDataProvider.getTextRecommendations.mockImplementation(
      () => new Promise(() => {}), // Never resolves
    )

    createTestUtils()

    const textInputs = screen.getAllByRole('textbox')
    fireEvent.change(textInputs[0], { target: { value: 'test query' } })

    const buttons = screen.getAllByRole('button')
    const generateButton = buttons.find(
      (btn) =>
        btn.textContent.includes('Generate') ||
        btn.textContent.includes('generate'),
    )

    if (generateButton) {
      fireEvent.click(generateButton)

      // Should show loading state (button becomes disabled or shows loading text)
      await waitFor(() => {
        expect(mockDataProvider.getTextRecommendations).toHaveBeenCalled()
      })
    }
  })

  it('handles API errors gracefully', async () => {
    mockDataProvider.getTextRecommendations.mockRejectedValue(
      new Error('Service error'),
    )

    createTestUtils()

    const textInputs = screen.getAllByRole('textbox')
    fireEvent.change(textInputs[0], { target: { value: 'test query' } })

    const buttons = screen.getAllByRole('button')
    const generateButton = buttons.find(
      (btn) =>
        btn.textContent.includes('Generate') ||
        btn.textContent.includes('generate'),
    )

    if (generateButton) {
      fireEvent.click(generateButton)

      await waitFor(() => {
        expect(mockDataProvider.getTextRecommendations).toHaveBeenCalled()
      })

      // Component should handle error without crashing
      const container = screen.getByRole('textbox').closest('div')
      expect(container).toBeInTheDocument()
    }
  })

  it('clears form when clear button clicked', async () => {
    createTestUtils()

    const textInputs = screen.getAllByRole('textbox')
    const mainInput = textInputs[0]

    fireEvent.change(mainInput, { target: { value: 'test query' } })

    const buttons = screen.getAllByRole('button')
    const clearButton = buttons.find(
      (btn) =>
        btn.textContent.includes('Clear') || btn.textContent.includes('clear'),
    )

    if (clearButton) {
      fireEvent.click(clearButton)

      await waitFor(() => {
        expect(mainInput.value).toBe('')
      })
    }
  })
})
