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
import BatchEmbeddingPanel from './BatchEmbeddingPanel'

describe('BatchEmbeddingPanel', () => {
  let mockDataProvider

  beforeEach(() => {
    mockDataProvider = {
      startBatchEmbedding: vi.fn(),
      getBatchEmbeddingProgress: vi.fn(),
      cancelBatchEmbedding: vi.fn(),
      getGpuSettings: vi.fn().mockResolvedValue({
        data: {
          maxGpuMemoryGb: 9,
          precision: 'fp16',
          enableCpuOffload: true,
          device: 'auto',
          estimatedVramGb: 9,
        },
      }),
      updateGpuSettings: vi.fn().mockResolvedValue({
        data: {
          status: 'restarting',
          settings: {
            maxGpuMemoryGb: 9,
            precision: 'fp16',
            enableCpuOffload: true,
            device: 'auto',
            estimatedVramGb: 9,
          },
        },
      }),
    }

    // Mock localStorage for admin check
    Storage.prototype.getItem = vi.fn((key) => {
      if (key === 'role') return 'admin'
      return null
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  const createTestUtils = () =>
    render(
      <DataProviderContext.Provider value={mockDataProvider}>
        <TestContext
          initialState={{
            admin: { resources: {}, ui: { optimistic: false } },
          }}
        >
          <BatchEmbeddingPanel />
        </TestContext>
      </DataProviderContext.Provider>
    )

  it('renders without crashing', () => {
    const { container } = createTestUtils()
    expect(container).toBeInTheDocument()
  })

  it('shows start button when not running', () => {
    createTestUtils()
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThan(0)
  })

  it('opens configuration dialog when start clicked', async () => {
    createTestUtils()

    const startButton = screen.getByText(/Start Re-embedding/i)
    fireEvent.click(startButton)

    await waitFor(() => {
      // Dialog should open, look for dialog by role
      const dialogs = screen.queryAllByRole('dialog')
      expect(dialogs.length).toBeGreaterThan(0)
    })
  })

  it('shows model selection checkboxes in dialog', async () => {
    createTestUtils()

    const startButton = screen.getByText(/Start Re-embedding/i)
    fireEvent.click(startButton)

    await waitFor(() => {
      const checkboxes = screen.getAllByRole('checkbox')
      // Should have at least 3 model checkboxes
      expect(checkboxes.length).toBeGreaterThanOrEqual(3)
    })
  })

  it('starts batch job with selected models', async () => {
    mockDataProvider.startBatchEmbedding.mockResolvedValue({
      data: { status: 'started', total_tracks: 100 },
    })

    createTestUtils()

    // Open dialog
    const startButton = screen.getByText(/Start Re-embedding/i)
    fireEvent.click(startButton)

    await waitFor(() => {
      // Find and click start job button in dialog
      const allButtons = screen.getAllByRole('button')
      const startJobButton = allButtons.find(btn =>
        btn.textContent.includes('Start') || btn.textContent.includes('start')
      )
      if (startJobButton) {
        fireEvent.click(startJobButton)
      }
    })

    await waitFor(() => {
      expect(mockDataProvider.startBatchEmbedding).toHaveBeenCalled()
    })
  })

  it('polls for progress after job starts', async () => {
    mockDataProvider.startBatchEmbedding.mockResolvedValue({
      data: { status: 'started', total_tracks: 100 },
    })

    mockDataProvider.getBatchEmbeddingProgress.mockResolvedValue({
      data: {
        status: 'running',
        total_tracks: 100,
        processed_tracks: 50,
        progress_percent: 50,
      },
    })

    createTestUtils()

    // Start job (simplified)
    const startButton = screen.getByText(/Start Re-embedding/i)
    fireEvent.click(startButton)

    await waitFor(() => {
      const allButtons = screen.getAllByRole('button')
      const startJobButton = allButtons[allButtons.length - 1]
      fireEvent.click(startJobButton)
    })

    // Should poll for progress
    await waitFor(
      () => {
        expect(mockDataProvider.getBatchEmbeddingProgress).toHaveBeenCalled()
      },
      { timeout: 2000 }
    )
  })

  it('handles job completion', async () => {
    mockDataProvider.startBatchEmbedding.mockResolvedValue({
      data: { status: 'started', total_tracks: 10 },
    })

    mockDataProvider.getBatchEmbeddingProgress.mockResolvedValue({
      data: {
        status: 'completed',
        total_tracks: 10,
        processed_tracks: 10,
        progress_percent: 100,
      },
    })

    createTestUtils()

    const startButton = screen.getByText(/Start Re-embedding/i)
    fireEvent.click(startButton)

    await waitFor(() => {
      const allButtons = screen.getAllByRole('button')
      fireEvent.click(allButtons[allButtons.length - 1])
    })

    await waitFor(
      () => {
        expect(mockDataProvider.getBatchEmbeddingProgress).toHaveBeenCalled()
      },
      { timeout: 2000 }
    )
  })
})
