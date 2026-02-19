import React, { useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  CircularProgress,
  FormControl,
  FormHelperText,
  IconButton,
  InputLabel,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Select,
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

const TEXT_TARGET_OPTIONS = [
  {
    value: 'lyrics',
    label: 'Lyrics',
    description: 'Lyric similarity search',
  },
  {
    value: 'description',
    label: 'Description',
    description: 'Generated description similarity search',
  },
]

const TextPlaylistGenerator = ({ onPlaylistGenerated }) => {
  const classes = useStyles()
  const translate = useTranslate()
  const dataProvider = useDataProvider()
  const notify = useNotify()

  const [textQuery, setTextQuery] = useState('')
  const [textTargets, setTextTargets] = useState(['lyrics', 'description'])
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

  const handleGenerate = async () => {
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
        textTargets,
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
      const message =
        err?.body?.message ||
        err?.message ||
        translate('pages.explore.textGenerator.error', {
          _: 'Failed to generate playlist from text query',
        })
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
              _: "Describe the music you want and we'll find matching tracks in your library.",
            })}
          </Typography>

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
              rows={2}
              variant="outlined"
              fullWidth
              disabled={loading}
            />
          </Box>

          <Box className={classes.formRow}>
            <FormControl className={classes.modelSelect} variant="outlined">
              <InputLabel id="text-targets-label">
                {translate('pages.explore.textGenerator.targets', {
                  _: 'Text targets',
                })}
              </InputLabel>
              <Select
                labelId="text-targets-label"
                multiple
                value={textTargets}
                onChange={(event) => {
                  const value = event.target.value
                  const next = Array.isArray(value) ? value : []
                  setTextTargets(next.length > 0 ? next : ['lyrics'])
                }}
                label={translate('pages.explore.textGenerator.targets', {
                  _: 'Text targets',
                })}
                disabled={loading}
                renderValue={(selected) =>
                  (Array.isArray(selected) ? selected : [])
                    .map((item) => {
                      const option = TEXT_TARGET_OPTIONS.find(
                        (target) => target.value === item,
                      )
                      return option?.label || item
                    })
                    .join(', ')
                }
              >
                {TEXT_TARGET_OPTIONS.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    <Checkbox checked={textTargets.indexOf(option.value) > -1} />
                    <ListItemText
                      primary={option.label}
                      secondary={option.description}
                    />
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                {translate('pages.explore.textGenerator.targetsHelper', {
                  _: 'Choose one or both text spaces for recommendation.',
                })}
              </FormHelperText>
            </FormControl>

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
              disabled={loading || !textQuery.trim()}
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
                  {result.warnings.join(', ')}
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
