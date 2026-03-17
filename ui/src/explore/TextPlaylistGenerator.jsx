import React, { useState } from 'react'
import {
  Box,
  Button,
  Card,
  CircularProgress,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Slider,
  TextField,
  Typography,
  Chip,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import { useDataProvider, useNotify, useTranslate } from 'react-admin'
import PlaylistAddIcon from '@material-ui/icons/PlaylistAdd'
import DeleteIcon from '@material-ui/icons/Delete'
import AddIcon from '@material-ui/icons/Add'
import MusicNoteIcon from '@material-ui/icons/MusicNote'
import {
  formatRecommendationError,
  healthMessageForMode,
  isRecommendationModeAvailable,
  sanitizeRecommendationWarning,
} from '../recommendationHealth'

const useStyles = makeStyles((theme) => ({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(3),
  },
  card: {
    padding: theme.spacing(3),
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(2),
  },
  formRow: {
    display: 'flex',
    gap: theme.spacing(2),
    alignItems: 'flex-start',
  },
  textInput: {
    flex: 1,
  },
  modelSelect: {
    minWidth: 150,
  },
  negativePromptRow: {
    display: 'flex',
    gap: theme.spacing(1),
    alignItems: 'center',
  },
  sliderContainer: {
    paddingLeft: theme.spacing(2),
    paddingRight: theme.spacing(2),
  },
  trackList: {
    maxHeight: 320,
    overflowY: 'auto',
    borderRadius: theme.shape.borderRadius,
    border: `1px solid ${theme.palette.divider}`,
  },
  trackItem: {
    borderBottom: `1px solid ${theme.palette.divider}`,
    '&:last-child': {
      borderBottom: 'none',
    },
  },
  buttonRow: {
    display: 'flex',
    gap: theme.spacing(1),
    alignItems: 'center',
  },
  warning: {
    color: theme.palette.warning.main,
  },
  error: {
    color: theme.palette.error.main,
  },
  modelChips: {
    display: 'flex',
    gap: theme.spacing(1),
    flexWrap: 'wrap',
  },
}))

const TextPlaylistGenerator = ({
  onPlaylistGenerated,
  health,
  healthLoading,
}) => {
  const classes = useStyles()
  const translate = useTranslate()
  const dataProvider = useDataProvider()
  const notify = useNotify()

  const [textQuery, setTextQuery] = useState('')
  const [negativePrompts, setNegativePrompts] = useState([])
  const [negativePenalty, setNegativePenalty] = useState(0.85)
  const [limit, setLimit] = useState(25)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleAddNegativePrompt = () => {
    setNegativePrompts([...negativePrompts, ''])
  }

  const handleRemoveNegativePrompt = (index) => {
    setNegativePrompts(negativePrompts.filter((_, i) => i !== index))
  }

  const handleUpdateNegativePrompt = (index, value) => {
    const updated = [...negativePrompts]
    updated[index] = value
    setNegativePrompts(updated)
  }

  const textModeAvailable =
    !health || isRecommendationModeAvailable(health, 'text')

  const handleGenerate = async () => {
    if (!textModeAvailable) {
      const message = healthMessageForMode(health, 'text', translate)
      setError(message)
      notify(message, { type: 'warning' })
      return
    }
    if (!textQuery.trim()) {
      notify('Please enter a text description', { type: 'warning' })
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const options = {
        text: textQuery,
        models: ['muq_mulan'],
        limit,
        negativePrompts: negativePrompts
          .map((p) => p.trim())
          .filter((p) => p !== ''),
        negativePromptPenalty: negativePenalty,
      }

      const { data } = await dataProvider.getTextRecommendations(options)

      if (data && data.tracks) {
        setResult(data)
        if (onPlaylistGenerated) {
          onPlaylistGenerated(data)
        }
        notify(
          translate('pages.explore.textGenerator.success', {
            _: `Generated ${data.tracks.length} tracks`,
            count: data.tracks.length,
          }),
          { type: 'success' },
        )
      } else {
        throw new Error('No tracks returned')
      }
    } catch (err) {
      const message = formatRecommendationError(
        err,
        translate,
        translate('pages.explore.textGenerator.error', {
          _: 'Failed to generate playlist from text query',
        }),
      )
      setError(message)
      notify(message, { type: 'error' })
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setTextQuery('')
    setNegativePrompts([])
    setResult(null)
    setError(null)
  }

  return (
    <Box className={classes.container}>
      <Card className={classes.card}>
        <Box className={classes.section}>
          <Typography variant="h6" gutterBottom>
            <MusicNoteIcon
              style={{ verticalAlign: 'middle', marginRight: 8 }}
            />
            {translate('pages.explore.textGenerator.title', {
              _: 'Generate Playlist from Text',
            })}
          </Typography>

          <Typography variant="body2" color="textSecondary">
            {translate('pages.explore.textGenerator.description', {
              _: 'Describe the music you want and muq_mulan will find matching tracks in your library.',
            })}
          </Typography>

          {!healthLoading && !textModeAvailable && (
            <Typography variant="body2" className={classes.warning}>
              {healthMessageForMode(health, 'text', translate)}
            </Typography>
          )}

          <Box className={classes.formRow}>
            <TextField
              className={classes.textInput}
              label={translate('pages.explore.textGenerator.queryLabel', {
                _: 'Describe the music',
              })}
              placeholder={translate(
                'pages.explore.textGenerator.queryPlaceholder',
                {
                  _: 'e.g., upbeat rock with guitar solos, chill jazz for studying...',
                },
              )}
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              multiline
              minRows={2}
              variant="outlined"
              fullWidth
              disabled={loading}
            />
          </Box>

          <Box className={classes.formRow}>
            <TextField
              type="number"
              label={translate('pages.explore.textGenerator.limit', {
                _: 'Track Limit',
              })}
              value={limit}
              onChange={(e) =>
                setLimit(
                  Math.max(1, Math.min(100, parseInt(e.target.value) || 25)),
                )
              }
              variant="outlined"
              inputProps={{ min: 1, max: 100 }}
              style={{ width: 120 }}
              disabled={loading}
            />
          </Box>

          {/* Negative Prompts Section */}
          <Box className={classes.section}>
            <Box
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <Typography variant="subtitle2">
                {translate('pages.explore.textGenerator.negativePrompts', {
                  _: 'Negative Prompts (Optional)',
                })}
              </Typography>
              <Button
                size="small"
                startIcon={<AddIcon />}
                onClick={handleAddNegativePrompt}
                disabled={loading}
              >
                {translate('pages.explore.textGenerator.addNegative', {
                  _: 'Add',
                })}
              </Button>
            </Box>

            {negativePrompts.length > 0 && (
              <>
                <Typography variant="caption" color="textSecondary">
                  {translate(
                    'pages.explore.textGenerator.negativeDescription',
                    {
                      _: 'Describe music styles to avoid',
                    },
                  )}
                </Typography>

                {negativePrompts.map((prompt, index) => (
                  <Box key={index} className={classes.negativePromptRow}>
                    <TextField
                      value={prompt}
                      onChange={(e) =>
                        handleUpdateNegativePrompt(index, e.target.value)
                      }
                      placeholder={translate(
                        'pages.explore.textGenerator.negativePlaceholder',
                        {
                          _: 'e.g., slow ballads, acoustic guitar...',
                        },
                      )}
                      variant="outlined"
                      size="small"
                      fullWidth
                      disabled={loading}
                    />
                    <IconButton
                      size="small"
                      onClick={() => handleRemoveNegativePrompt(index)}
                      disabled={loading}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                ))}

                <Box className={classes.sliderContainer}>
                  <Typography variant="caption" gutterBottom>
                    {translate('pages.explore.textGenerator.penalty', {
                      _: 'Negative Prompt Penalty',
                    })}
                  </Typography>
                  <Slider
                    value={negativePenalty}
                    onChange={(e, val) => setNegativePenalty(val)}
                    min={0.3}
                    max={1.0}
                    step={0.05}
                    marks={[
                      {
                        value: 0.3,
                        label: translate(
                          'pages.explore.textGenerator.penaltyStrong',
                          { _: 'Strong' },
                        ),
                      },
                      {
                        value: 0.85,
                        label: translate(
                          'pages.explore.textGenerator.penaltyDefault',
                          { _: 'Default' },
                        ),
                      },
                      {
                        value: 1.0,
                        label: translate(
                          'pages.explore.textGenerator.penaltyNone',
                          { _: 'None' },
                        ),
                      },
                    ]}
                    valueLabelDisplay="auto"
                    disabled={loading}
                  />
                </Box>
              </>
            )}
          </Box>

          {/* Action Buttons */}
          <Box className={classes.buttonRow}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleGenerate}
              disabled={loading || !textQuery.trim() || !textModeAvailable}
              startIcon={
                loading ? <CircularProgress size={20} /> : <PlaylistAddIcon />
              }
            >
              {loading
                ? translate('pages.explore.textGenerator.generating', {
                    _: 'Generating...',
                  })
                : translate('pages.explore.textGenerator.generate', {
                    _: 'Generate Playlist',
                  })}
            </Button>
            <Button variant="outlined" onClick={handleClear} disabled={loading}>
              {translate('pages.explore.textGenerator.clear', { _: 'Clear' })}
            </Button>
          </Box>

          {/* Error Message */}
          {error && (
            <Typography variant="body2" className={classes.error}>
              {error}
            </Typography>
          )}

          {/* Results */}
          {result && result.tracks && result.tracks.length > 0 && (
            <Box className={classes.section}>
              <Typography variant="subtitle1">
                {translate('pages.explore.textGenerator.results', {
                  _: 'Generated Playlist',
                })}{' '}
                ({result.tracks.length}{' '}
                {translate('pages.explore.textGenerator.tracks', {
                  _: 'tracks',
                })}
                )
              </Typography>

              {result.warnings && result.warnings.length > 0 && (
                <Typography variant="caption" className={classes.warning}>
                  {result.warnings
                    .map((warning) =>
                      sanitizeRecommendationWarning(warning, translate),
                    )
                    .join(', ')}
                </Typography>
              )}

              <List className={classes.trackList}>
                {result.tracks.map((track, index) => (
                  <ListItem key={track.id} className={classes.trackItem}>
                    <ListItemText
                      primary={`${index + 1}. ${track.title}`}
                      secondary={`${track.artist}${track.album ? ` • ${track.album}` : ''}`}
                    />
                    {track.models && track.models.length > 0 && (
                      <Box className={classes.modelChips}>
                        {track.models.map((model) => (
                          <Chip
                            key={model}
                            label={model.toUpperCase()}
                            size="small"
                          />
                        ))}
                      </Box>
                    )}
                  </ListItem>
                ))}
              </List>
            </Box>
          )}
        </Box>
      </Card>
    </Box>
  )
}

export default TextPlaylistGenerator
