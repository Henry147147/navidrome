import React from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography,
  Chip,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'

const useStyles = makeStyles((theme) => ({
  card: {
    padding: theme.spacing(1, 2, 2),
  },
  header: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(1),
    marginBottom: theme.spacing(2),
  },
  fieldGrid: {
    display: 'grid',
    gap: theme.spacing(3),
    gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
  },
  helperText: {
    color: theme.palette.text.secondary,
  },
  buttonRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(1),
    marginTop: theme.spacing(3),
  },
  loading: {
    display: 'flex',
    justifyContent: 'center',
    padding: theme.spacing(4, 0),
  },
  error: {
    color: theme.palette.error.main,
    marginBottom: theme.spacing(2),
  },
  chips: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(0.5),
  },
}))

const AutoPlaySettingsPanel = ({
  translate,
  draft,
  onFieldChange,
  onSave,
  onReset,
  saving,
  loading,
  error,
  playlists,
  modeOptions,
}) => {
  const classes = useStyles()

  if (loading) {
    return (
      <Card className={classes.card} variant="outlined">
        <CardContent className={classes.loading}>
          <CircularProgress size={28} />
        </CardContent>
      </Card>
    )
  }

  const handleModeChange = (event) => {
    onFieldChange('mode', event.target.value)
  }

  const handleBatchSizeChange = (event) => {
    const value = Number(event.target.value)
    onFieldChange('batchSize', Number.isNaN(value) ? draft.batchSize : value)
  }

  const handleDiversityChange = (event) => {
    const raw = event.target.value
    if (raw === '' || raw === null) {
      onFieldChange('diversityOverride', null)
      return
    }
    const value = Number(raw)
    onFieldChange('diversityOverride', Number.isNaN(value) ? null : value)
  }

  const handleTextPromptChange = (event) => {
    onFieldChange('textPrompt', event.target.value)
  }

  const handleExcludePlaylistsChange = (event) => {
    onFieldChange('excludePlaylistIds', event.target.value)
  }

  return (
    <Card className={classes.card} variant="outlined">
      <CardContent>
        <Box className={classes.header}>
          <Typography variant="h5">
            {translate('pages.autoplay.settings.title', {
              _: 'Auto Play settings',
            })}
          </Typography>
          <Typography variant="body2" className={classes.helperText}>
            {translate('pages.autoplay.settings.helper', {
              _: 'Choose your default Auto Play behaviour. You can override these on the Mix tab.',
            })}
          </Typography>
        </Box>

        {error && (
          <Typography variant="body2" className={classes.error}>
            {error}
          </Typography>
        )}

        <Box className={classes.fieldGrid}>
          <FormControl variant="outlined" fullWidth>
            <InputLabel id="autoplay-mode-label">
              {translate('pages.autoplay.settings.modeLabel', {
                _: 'Default mode',
              })}
            </InputLabel>
            <Select
              labelId="autoplay-mode-label"
              value={draft.mode}
              label={translate('pages.autoplay.settings.modeLabel', {
                _: 'Default mode',
              })}
              onChange={handleModeChange}
            >
              {modeOptions.map((option) => (
                <MenuItem value={option.value} key={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            variant="outlined"
            type="number"
            inputProps={{ min: 5, max: 50 }}
            label={translate('pages.autoplay.settings.batchLabel', {
              _: 'Batch size',
            })}
            helperText={translate('pages.autoplay.settings.batchHelper', {
              _: 'How many tracks to request at once.',
            })}
            value={draft.batchSize}
            onChange={handleBatchSizeChange}
            fullWidth
          />

          <TextField
            variant="outlined"
            type="number"
            inputProps={{ step: 0.05, min: 0, max: 1 }}
            label={translate('pages.autoplay.settings.diversityLabel', {
              _: 'Diversity override',
            })}
            helperText={translate('pages.autoplay.settings.diversityHelper', {
              _: 'Optional value between 0 and 1. Leave blank to follow mix defaults.',
            })}
            value={draft.diversityOverride ?? ''}
            onChange={handleDiversityChange}
            fullWidth
          />

          <TextField
            variant="outlined"
            label={translate('pages.autoplay.settings.textPromptLabel', {
              _: 'Default text prompt',
            })}
            helperText={translate('pages.autoplay.settings.textPromptHelper', {
              _: 'Used when the Text mode is selected on the Mix tab.',
            })}
            value={draft.textPrompt}
            onChange={handleTextPromptChange}
            multiline
            rows={3}
            fullWidth
          />

          <FormControl variant="outlined" fullWidth>
            <InputLabel id="autoplay-exclude-playlists">
              {translate('pages.autoplay.settings.excludePlaylistsLabel', {
                _: 'Exclude playlists',
              })}
            </InputLabel>
            <Select
              labelId="autoplay-exclude-playlists"
              multiple
              value={draft.excludePlaylistIds}
              onChange={handleExcludePlaylistsChange}
              label={translate(
                'pages.autoplay.settings.excludePlaylistsLabel',
                {
                  _: 'Exclude playlists',
                },
              )}
              renderValue={(selected) => (
                <Box className={classes.chips}>
                  {selected.map((id) => {
                    const playlist = playlists.find((p) => p.id === id)
                    return (
                      <Chip
                        key={id}
                        size="small"
                        label={playlist ? playlist.name : id}
                      />
                    )
                  })}
                </Box>
              )}
            >
              {playlists.map((playlist) => (
                <MenuItem value={playlist.id} key={playlist.id}>
                  {playlist.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        <Box className={classes.buttonRow}>
          <Button
            variant="contained"
            color="primary"
            onClick={onSave}
            disabled={saving}
          >
            {saving
              ? translate('pages.autoplay.settings.saving', { _: 'Saving…' })
              : translate('pages.autoplay.settings.save', {
                  _: 'Save settings',
                })}
          </Button>
          <Button onClick={onReset} disabled={saving}>
            {translate('pages.autoplay.settings.reset', { _: 'Reset changes' })}
          </Button>
        </Box>
      </CardContent>
    </Card>
  )
}

export default AutoPlaySettingsPanel
