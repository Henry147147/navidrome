import React, { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  LinearProgress,
  Typography,
  List,
  ListItem,
  ListItemText,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import {
  useDataProvider,
  useNotify,
  useTranslate,
} from 'react-admin'
import PlayArrowIcon from '@material-ui/icons/PlayArrow'
import StopIcon from '@material-ui/icons/Stop'
import WarningIcon from '@material-ui/icons/Warning'
import CheckCircleIcon from '@material-ui/icons/CheckCircle'
import ErrorIcon from '@material-ui/icons/Error'

const useStyles = makeStyles((theme) => ({
  card: {
    marginTop: theme.spacing(2),
  },
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(2),
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(2),
  },
  progressContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(1),
  },
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
  },
  warning: {
    color: theme.palette.warning.main,
  },
  error: {
    color: theme.palette.error.main,
  },
  success: {
    color: theme.palette.success.main,
  },
  modelSelection: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(1),
    marginTop: theme.spacing(1),
  },
  dialogContent: {
    minWidth: 400,
  },
}))

const MODEL_OPTIONS = [
  { value: 'muq', label: 'MuQ (1536-dim)', description: 'MuQ-MuLan model' },
  { value: 'mert', label: 'MERT (76,800-dim)', description: 'MERT-v1-330M model' },
  { value: 'latent', label: 'Latent Space (576-dim)', description: 'Music2Latent model' },
]

const BatchEmbeddingPanel = () => {
  const classes = useStyles()
  const translate = useTranslate()
  const dataProvider = useDataProvider()
  const notify = useNotify()

  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(null)
  const [selectedModels, setSelectedModels] = useState(['muq', 'mert', 'latent'])
  const [clearExisting, setClearExisting] = useState(true)
  const [error, setError] = useState(null)

  // Poll for progress when job is running
  useEffect(() => {
    if (!isRunning) {
      return
    }

    const pollInterval = setInterval(async () => {
      try {
        const { data } = await dataProvider.getBatchEmbeddingProgress()
        setProgress(data)

        // Check if job completed or was cancelled
        if (data.status === 'completed' ||
            data.status === 'cancelled' ||
            data.status === 'completed_with_errors' ||
            data.status === 'failed') {
          setIsRunning(false)
          clearInterval(pollInterval)

          if (data.status === 'completed') {
            notify(
              translate('pages.settings.batchEmbedding.completed', {
                _: `Batch embedding completed: ${data.processed_tracks} tracks processed`,
                count: data.processed_tracks,
              }),
              'success'
            )
          } else if (data.status === 'completed_with_errors') {
            notify(
              translate('pages.settings.batchEmbedding.completedWithErrors', {
                _: `Batch embedding completed with ${data.failed_tracks} errors`,
                count: data.failed_tracks,
              }),
              'warning'
            )
          } else if (data.status === 'cancelled') {
            notify(
              translate('pages.settings.batchEmbedding.cancelled', {
                _: 'Batch embedding was cancelled',
              }),
              'info'
            )
          } else if (data.status === 'failed') {
            notify(
              translate('pages.settings.batchEmbedding.failed', {
                _: 'Batch embedding failed',
              }),
              'error'
            )
          }
        }
      } catch (err) {
        console.error('Failed to get batch progress:', err)
        setError(err.message || 'Failed to get progress')
        setIsRunning(false)
        clearInterval(pollInterval)
      }
    }, 1000) // Poll every second

    return () => clearInterval(pollInterval)
  }, [isRunning, dataProvider, notify, translate])

  const handleStart = useCallback(async () => {
    if (selectedModels.length === 0) {
      notify(
        translate('pages.settings.batchEmbedding.noModelsSelected', {
          _: 'Please select at least one model',
        }),
        'warning'
      )
      return
    }

    try {
      setError(null)
      const { data } = await dataProvider.startBatchEmbedding(
        selectedModels,
        clearExisting
      )

      if (data && data.status === 'started') {
        setIsRunning(true)
        setIsDialogOpen(false)
        notify(
          translate('pages.settings.batchEmbedding.started', {
            _: 'Batch embedding job started',
          }),
          'info'
        )
      } else {
        throw new Error('Unexpected response from server')
      }
    } catch (err) {
      console.error('Failed to start batch embedding:', err)
      const message =
        err?.body?.message ||
        err?.message ||
        translate('pages.settings.batchEmbedding.startError', {
          _: 'Failed to start batch embedding job',
        })
      setError(message)
      notify(message, 'error')
    }
  }, [selectedModels, clearExisting, dataProvider, notify, translate])

  const handleCancel = useCallback(async () => {
    try {
      await dataProvider.cancelBatchEmbedding()
      notify(
        translate('pages.settings.batchEmbedding.cancelling', {
          _: 'Cancelling batch embedding job...',
        }),
        'info'
      )
    } catch (err) {
      console.error('Failed to cancel batch embedding:', err)
      notify(
        translate('pages.settings.batchEmbedding.cancelError', {
          _: 'Failed to cancel batch embedding job',
        }),
        'error'
      )
    }
  }, [dataProvider, notify, translate])

  const handleModelToggle = (modelValue) => {
    if (selectedModels.includes(modelValue)) {
      setSelectedModels(selectedModels.filter(m => m !== modelValue))
    } else {
      setSelectedModels([...selectedModels, modelValue])
    }
  }

  const formatETA = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp * 1000)
    return date.toLocaleTimeString()
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className={classes.success} />
      case 'completed_with_errors':
        return <WarningIcon className={classes.warning} />
      case 'failed':
      case 'error':
        return <ErrorIcon className={classes.error} />
      default:
        return null
    }
  }

  return (
    <Card className={classes.card}>
      <CardContent>
        <Box className={classes.container}>
          <Box className={classes.header}>
            <Typography variant="h6">
              {translate('pages.settings.batchEmbedding.title', {
                _: 'Batch Re-embedding',
              })}
            </Typography>
          </Box>

          <Typography variant="body2" color="textSecondary">
            {translate('pages.settings.batchEmbedding.description', {
              _: 'Re-generate embeddings for all tracks in your library. This may take several hours for large libraries.',
            })}
          </Typography>

          {isRunning && progress && (
            <Box className={classes.progressContainer}>
              <Box className={classes.statusRow}>
                <Typography variant="body2">
                  {translate('pages.settings.batchEmbedding.status', {
                    _: 'Status',
                  })}
                  : {progress.status}
                </Typography>
                {getStatusIcon(progress.status)}
              </Box>

              <LinearProgress
                variant="determinate"
                value={progress.progress_percent || 0}
              />

              <Typography variant="body2">
                {translate('pages.settings.batchEmbedding.progress', {
                  _: 'Progress',
                })}
                : {progress.processed_operations || 0} / {progress.total_operations || 0}
                {' '}({Math.round(progress.progress_percent || 0)}%)
              </Typography>

              {progress.total_tracks && progress.processed_tracks > 0 && (
                <Typography variant="caption" color="textSecondary">
                  {translate('pages.settings.batchEmbedding.tracksProcessed', {
                    _: 'Tracks processed',
                  })}
                  : {progress.processed_tracks} / {progress.total_tracks}
                </Typography>
              )}

              {progress.current_model && (
                <Typography variant="caption" color="textSecondary">
                  {translate('pages.settings.batchEmbedding.currentModel', {
                    _: 'Current model',
                  })}
                  : {progress.current_model}
                </Typography>
              )}

              {progress.current_track && (
                <Typography variant="caption" color="textSecondary">
                  {translate('pages.settings.batchEmbedding.currentTrack', {
                    _: 'Current track',
                  })}
                  : {progress.current_track}
                </Typography>
              )}

              {progress.failed_tracks > 0 && (
                <Typography variant="caption" className={classes.warning}>
                  {translate('pages.settings.batchEmbedding.failed', {
                    _: 'Failed tracks',
                    count: progress.failed_tracks,
                  })}
                  : {progress.failed_tracks}
                </Typography>
              )}

              {progress.estimated_completion && (
                <Typography variant="caption" color="textSecondary">
                  {translate('pages.settings.batchEmbedding.eta', {
                    _: 'Estimated completion',
                  })}
                  : {formatETA(progress.estimated_completion)}
                </Typography>
              )}

              <Button
                variant="outlined"
                color="secondary"
                onClick={handleCancel}
                startIcon={<StopIcon />}
              >
                {translate('pages.settings.batchEmbedding.cancel', {
                  _: 'Cancel Job',
                })}
              </Button>
            </Box>
          )}

          {!isRunning && (
            <Button
              variant="contained"
              color="primary"
              onClick={() => setIsDialogOpen(true)}
              startIcon={<PlayArrowIcon />}
            >
              {translate('pages.settings.batchEmbedding.start', {
                _: 'Start Re-embedding',
              })}
            </Button>
          )}

          {error && (
            <Typography variant="body2" className={classes.error}>
              {error}
            </Typography>
          )}
        </Box>
      </CardContent>

      {/* Configuration Dialog */}
      <Dialog
        open={isDialogOpen}
        onClose={() => !isRunning && setIsDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {translate('pages.settings.batchEmbedding.dialogTitle', {
            _: 'Configure Batch Re-embedding',
          })}
        </DialogTitle>
        <DialogContent className={classes.dialogContent}>
          <Box className={classes.container}>
            <Typography variant="body2">
              {translate('pages.settings.batchEmbedding.selectModels', {
                _: 'Select embedding models to use:',
              })}
            </Typography>

            <Box className={classes.modelSelection}>
              {MODEL_OPTIONS.map((option) => (
                <FormControlLabel
                  key={option.value}
                  control={
                    <Checkbox
                      checked={selectedModels.includes(option.value)}
                      onChange={() => handleModelToggle(option.value)}
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2">{option.label}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {option.description}
                      </Typography>
                    </Box>
                  }
                />
              ))}
            </Box>

            <FormControlLabel
              control={
                <Checkbox
                  checked={clearExisting}
                  onChange={(e) => setClearExisting(e.target.checked)}
                />
              }
              label={translate('pages.settings.batchEmbedding.clearExisting', {
                _: 'Clear existing embeddings first',
              })}
            />

            <Box className={classes.warning} style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
              <WarningIcon />
              <Typography variant="body2">
                {translate('pages.settings.batchEmbedding.warning', {
                  _: 'This operation may take several hours for large libraries. The recommendation system will be unavailable during this time.',
                })}
              </Typography>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsDialogOpen(false)} disabled={isRunning}>
            {translate('ra.action.cancel', { _: 'Cancel' })}
          </Button>
          <Button
            onClick={handleStart}
            color="primary"
            variant="contained"
            disabled={selectedModels.length === 0}
          >
            {translate('pages.settings.batchEmbedding.startJob', {
              _: 'Start Job',
            })}
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  )
}

export default BatchEmbeddingPanel
