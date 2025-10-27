import React from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Slider,
  Typography,
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
  sliderList: {
    display: 'grid',
    gap: theme.spacing(3),
  },
  sliderGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(1),
  },
  sliderHeader: {
    display: 'flex',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: theme.spacing(1),
  },
  helperText: {
    color: theme.palette.text.secondary,
  },
  error: {
    color: theme.palette.error.main,
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
  serverError: {
    color: theme.palette.error.main,
    marginBottom: theme.spacing(2),
  },
}))

const ExploreSettingsPanel = ({
  translate,
  draft,
  errors,
  onFieldChange,
  onReset,
  onSave,
  saving,
  dirty,
  loading,
  serverError,
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

  const sliderConfigs = [
    {
      key: 'mixLength',
      min: 10,
      max: 100,
      step: 1,
      label: translate('pages.explore.settings.mixLength', {
        _: 'Default mix length',
      }),
      helper: translate('pages.explore.settings.mixLengthHelper', {
        _: 'Applies to all automatic mix generators.',
      }),
      format: (value) => `${value} ${translate('pages.explore.settings.tracksLabel', { _: 'tracks' })}`,
    },
    {
      key: 'baseDiversity',
      min: 0,
      max: 1,
      step: 0.05,
      label: translate('pages.explore.settings.baseDiversity', {
        _: 'Primary exploration',
      }),
      helper: translate('pages.explore.settings.baseDiversityHelper', {
        _: 'Lower values stay close to your tastes; higher values introduce variety.',
      }),
      format: (value) => `${Math.round(value * 100)}%`,
    },
    {
      key: 'discoveryExploration',
      min: 0.3,
      max: 1,
      step: 0.05,
      label: translate('pages.explore.settings.discoveryExploration', {
        _: 'Discovery exploration',
      }),
      helper: translate('pages.explore.settings.discoveryExplorationHelper', {
        _: 'Controls how adventurous the discovery mix should be.',
      }),
      format: (value) => `${Math.round(value * 100)}%`,
    },
    {
      key: 'seedRecencyWindowDays',
      min: 7,
      max: 120,
      step: 1,
      label: translate('pages.explore.settings.recencyWindow', {
        _: 'Recent listens window',
      }),
      helper: translate('pages.explore.settings.recencyWindowHelper', {
        _: 'Only plays within this many days are used as recent seeds.',
      }),
      format: (value) =>
        translate('pages.explore.settings.daysLabel', {
          _: '%{value} days',
          value,
        }),
    },
    {
      key: 'favoritesBlendWeight',
      min: 0.1,
      max: 1,
      step: 0.05,
      label: translate('pages.explore.settings.favoritesWeight', {
        _: 'Favorites blend weight',
      }),
      helper: translate('pages.explore.settings.favoritesWeightHelper', {
        _: 'Balances starred tracks against recent plays in blended mixes.',
      }),
      format: (value) => `${Math.round(value * 100)}%`,
    },
  ]

  return (
    <Card className={classes.card} variant="outlined">
      <CardContent>
        <Box className={classes.header}>
          <Typography variant="h5">
            {translate('pages.explore.settings.title', {
              _: 'Recommendation settings',
            })}
          </Typography>
          <Typography variant="body2" className={classes.helperText}>
            {translate('pages.explore.settings.subtitle', {
              _: 'Tune the hyperparameters that power Explore mixes.',
            })}
          </Typography>
        </Box>

        {serverError && (
          <Typography variant="body2" className={classes.serverError}>
            {serverError}
          </Typography>
        )}

        <Box className={classes.sliderList}>
          {sliderConfigs.map((config) => {
            const value = draft[config.key]
            const handleChange = (_, nextValue) => {
              const numericValue = Array.isArray(nextValue) ? nextValue[0] : nextValue
              onFieldChange(config.key, numericValue)
            }
            return (
              <Box key={config.key} className={classes.sliderGroup}>
                <Box className={classes.sliderHeader}>
                  <Typography variant="subtitle1">{config.label}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    {config.format(value)}
                  </Typography>
                </Box>
                <Slider
                  value={value}
                  min={config.min}
                  max={config.max}
                  step={config.step}
                  onChange={handleChange}
                  valueLabelDisplay="auto"
                  valueLabelFormat={config.format}
                />
                <Typography variant="body2" className={classes.helperText}>
                  {config.helper}
                </Typography>
                {errors?.[config.key] && (
                  <Typography variant="caption" className={classes.error}>
                    {errors[config.key]}
                  </Typography>
                )}
              </Box>
            )
          })}
        </Box>

        <Box className={classes.buttonRow}>
          <Button
            variant="contained"
            color="primary"
            onClick={onSave}
            disabled={!dirty || saving}
          >
            {saving ? <CircularProgress size={18} color="inherit" /> :
              translate('pages.explore.settings.saveButton', { _: 'Save settings' })}
          </Button>
          <Button onClick={onReset} disabled={!dirty || saving}>
            {translate('pages.explore.settings.resetButton', { _: 'Reset' })}
          </Button>
        </Box>
      </CardContent>
    </Card>
  )
}

export default ExploreSettingsPanel
