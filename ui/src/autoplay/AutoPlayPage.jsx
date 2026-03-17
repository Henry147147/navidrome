import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  IconButton,
  InputAdornment,
  List,
  ListItem,
  ListItemText,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
  Chip,
} from '@material-ui/core'
import ListItemSecondaryAction from '@material-ui/core/ListItemSecondaryAction'
import { makeStyles } from '@material-ui/core/styles'
import { Title, useDataProvider, useNotify, useTranslate } from 'react-admin'
import { useDispatch, useSelector } from 'react-redux'
import SearchIcon from '@material-ui/icons/Search'
import PlayArrowIcon from '@material-ui/icons/PlayArrow'
import RefreshIcon from '@material-ui/icons/Refresh'
import ThumbUpAltOutlinedIcon from '@material-ui/icons/ThumbUpAltOutlined'
import ThumbDownAltOutlinedIcon from '@material-ui/icons/ThumbDownAltOutlined'
import ClearIcon from '@material-ui/icons/Clear'
import AutoPlaySettingsPanel from './AutoPlaySettingsPanel'
import {
  addTracks,
  clearQueue,
  markAutoPlayTracksRequested,
  playTracks,
  resetAutoPlayRuntime,
  setAutoPlayEnabled,
  syncAutoPlaySettings,
  toggleAutoPlayFeedback,
} from '../actions'
import {
  formatRecommendationError,
  healthMessageForMode,
  isRecommendationModeAvailable,
  sanitizeRecommendationWarning,
  useRecommendationHealth,
} from '../recommendationHealth'
import { normalizeAutoPlaySettings, refillAutoPlayQueue } from './runtime'

const DEFAULT_SETTINGS = {
  enabled: false,
  mode: 'recent',
  textPrompt: '',
  excludePlaylistIds: [],
  batchSize: 5,
  diversityOverride: null,
}

const AUTO_MODE_OPTIONS = (translate) => [
  {
    value: 'recent',
    label: translate('pages.autoplay.modes.recent', { _: 'Recent listens' }),
  },
  {
    value: 'favorites',
    label: translate('pages.autoplay.modes.favorites', { _: 'Liked songs' }),
  },
  {
    value: 'all',
    label: translate('pages.autoplay.modes.all', { _: 'Holistic mix' }),
  },
  {
    value: 'discovery',
    label: translate('pages.autoplay.modes.discovery', { _: 'Discovery' }),
  },
  {
    value: 'text',
    label: translate('pages.autoplay.modes.text', { _: 'Text prompt' }),
  },
  {
    value: 'custom',
    label: translate('pages.autoplay.modes.custom', { _: 'Specific song' }),
  },
]

const useStyles = makeStyles((theme) => ({
  root: {
    marginTop: theme.spacing(2),
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(3),
  },
  controlsCard: {
    padding: theme.spacing(2.5),
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(2),
  },
  controlGrid: {
    display: 'grid',
    gap: theme.spacing(2),
    gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
  },
  seedSearchResults: {
    border: `1px solid ${theme.palette.divider}`,
    borderRadius: theme.shape.borderRadius,
    maxHeight: 240,
    overflowY: 'auto',
  },
  queueCard: {
    padding: theme.spacing(2.5),
  },
  queueList: {
    maxHeight: 380,
    overflowY: 'auto',
  },
  modeSummary: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(1),
  },
  modeSummaryLine: {
    fontWeight: theme.typography.fontWeightMedium,
  },
  exclusionHint: {
    color: theme.palette.text.secondary,
  },
  warning: {
    color: theme.palette.warning.main,
  },
  actionRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(1.5),
    alignItems: 'center',
    marginTop: theme.spacing(2),
  },
  tabs: {
    alignSelf: 'flex-start',
  },
  chipRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(1),
  },
  fullWidthGridItem: {
    gridColumn: '1 / -1',
  },
  primaryActionButton: {
    minWidth: 200,
  },
  statusBanner: {
    padding: theme.spacing(1.5, 2),
    borderLeft: `4px solid ${theme.palette.warning.main}`,
  },
}))

const AutoPlayPage = () => {
  const classes = useStyles()
  const translate = useTranslate()
  const notify = useNotify()
  const dataProvider = useDataProvider()
  const dispatch = useDispatch()
  const player = useSelector((state) => state.player)
  const autoplay = useSelector((state) => state.autoplay)
  const { health: recommendationHealth, loading: recommendationHealthLoading } =
    useRecommendationHealth(dataProvider)

  const [tab, setTab] = useState(0)
  const [settingsLoading, setSettingsLoading] = useState(true)
  const [settingsError, setSettingsError] = useState('')
  const [playlists, setPlaylists] = useState([])
  const [playlistsLoading, setPlaylistsLoading] = useState(true)
  const [savingSettings, setSavingSettings] = useState(false)
  const [settingsDraft, setSettingsDraft] = useState(DEFAULT_SETTINGS)
  const [savedSettings, setSavedSettings] = useState(DEFAULT_SETTINGS)
  const [sessionOptions, setSessionOptions] = useState(DEFAULT_SETTINGS)
  const [fetching, setFetching] = useState(false)
  const fetchingRef = useRef(false)

  const [seedQuery, setSeedQuery] = useState('')
  const [seedLoading, setSeedLoading] = useState(false)
  const [seedResults, setSeedResults] = useState([])
  const [selectedSeed, setSelectedSeed] = useState(null)

  useEffect(() => {
    let mounted = true
    setSettingsLoading(true)
    setSettingsError('')

    dataProvider
      .getAutoPlaySettings()
      .then(({ data }) => {
        if (!mounted) {
          return
        }
        const normalized = normalizeAutoPlaySettings({
          ...DEFAULT_SETTINGS,
          ...data,
        })
        setSettingsDraft(normalized)
        setSessionOptions(normalized)
        setSavedSettings(normalized)
        dispatch(syncAutoPlaySettings(normalized))
      })
      .catch((error) => {
        if (mounted) {
          setSettingsError(error?.message || 'Failed to load settings')
        }
      })
      .finally(() => {
        if (mounted) {
          setSettingsLoading(false)
        }
      })

    setPlaylistsLoading(true)
    dataProvider
      .getList('playlist', {
        pagination: { page: 1, perPage: 200 },
        sort: { field: 'name', order: 'ASC' },
      })
      .then(({ data }) => {
        if (mounted) {
          setPlaylists(data || [])
        }
      })
      .catch(() => {
        if (mounted) {
          setPlaylists([])
        }
      })
      .finally(() => {
        if (mounted) {
          setPlaylistsLoading(false)
        }
      })

    return () => {
      mounted = false
    }
  }, [dataProvider, dispatch])

  const handleTabChange = (_, value) => {
    setTab(value)
  }

  const updateSessionOption = (key, value) => {
    setSessionOptions((prev) => ({ ...prev, [key]: value }))
  }

  const updateSettingsDraft = (key, value) => {
    setSettingsDraft((prev) => ({ ...prev, [key]: value }))
  }

  const resetFeedback = () => {
    dispatch(resetAutoPlayRuntime())
  }

  const handleSaveSettings = () => {
    setSavingSettings(true)
    const payload = {
      ...settingsDraft,
      enabled: Boolean(autoplay?.enabled),
      diversityOverride:
        settingsDraft.diversityOverride === null ||
        settingsDraft.diversityOverride === ''
          ? null
          : settingsDraft.diversityOverride,
    }
    payload.batchSize = DEFAULT_SETTINGS.batchSize
    dataProvider
      .updateAutoPlaySettings(payload)
      .then(({ data }) => {
        const normalized = normalizeAutoPlaySettings({
          ...DEFAULT_SETTINGS,
          ...data,
        })
        setSettingsDraft(normalized)
        setSessionOptions((prev) => ({ ...prev, ...normalized }))
        setSavedSettings(normalized)
        dispatch(syncAutoPlaySettings(normalized))
        notify('pages.autoplay.settings.saved', { type: 'info' })
      })
      .catch(() => {
        notify('pages.autoplay.settings.serverError', { type: 'warning' })
      })
      .finally(() => setSavingSettings(false))
  }

  useEffect(() => {
    if (!seedQuery.trim()) {
      setSeedResults([])
      setSeedLoading(false)
      return
    }
    setSeedLoading(true)
    const timer = setTimeout(() => {
      dataProvider
        .getList('song', {
          pagination: { page: 1, perPage: 15 },
          sort: { field: 'playDate', order: 'DESC' },
          filter: { q: seedQuery.trim() },
        })
        .then(({ data }) => {
          setSeedResults(data || [])
        })
        .catch(() => {
          setSeedResults([])
        })
        .finally(() => setSeedLoading(false))
    }, 250)
    return () => clearTimeout(timer)
  }, [seedQuery, dataProvider])

  useEffect(() => {
    if (sessionOptions.mode !== 'custom') {
      setSelectedSeed(null)
      setSeedResults([])
      setSeedQuery('')
    }
  }, [sessionOptions.mode])

  const buildExcludeIds = useCallback(() => {
    const combined = new Set([
      ...(autoplay?.playedTrackIds || []),
      ...(autoplay?.requestedTrackIds || []),
      ...(autoplay?.negativeTrackIds || []),
    ])
    return Array.from(combined)
  }, [autoplay])

  const ensureUniquePositive = useCallback(() => {
    return Array.from(new Set(autoplay?.positiveTrackIds || []))
  }, [autoplay])

  const fetchRecommendations = useCallback(
    async (options = {}) => {
      if (fetchingRef.current) {
        return
      }
      fetchingRef.current = true
      setFetching(true)

      const mode = options.mode || sessionOptions.mode || DEFAULT_SETTINGS.mode
      if (!isRecommendationModeAvailable(recommendationHealth, mode)) {
        notify(healthMessageForMode(recommendationHealth, mode, translate), {
          type: 'warning',
        })
        fetchingRef.current = false
        setFetching(false)
        return
      }
      const excludePlaylistIds =
        options.excludePlaylistIds || sessionOptions.excludePlaylistIds || []
      const excludeTrackIds = buildExcludeIds()
      const positiveIds = ensureUniquePositive()

      const payloadBase = {
        limit: sessionOptions.batchSize || DEFAULT_SETTINGS.batchSize,
        excludeTrackIds,
        excludePlaylistIds,
        positiveTrackIds: positiveIds,
        negativeTrackIds: autoplay?.negativeTrackIds || [],
      }
      if (
        sessionOptions.diversityOverride !== null &&
        sessionOptions.diversityOverride !== undefined &&
        sessionOptions.diversityOverride !== ''
      ) {
        payloadBase.diversity = Number(sessionOptions.diversityOverride)
      }

      const determineRequest = async () => {
        switch (mode) {
          case 'recent':
            return dataProvider.getRecentRecommendations(payloadBase)
          case 'favorites':
            return dataProvider.getFavoriteRecommendations(payloadBase)
          case 'all':
            return dataProvider.getAllRecommendations(payloadBase)
          case 'discovery':
            return dataProvider.getDiscoveryRecommendations(payloadBase)
          case 'text': {
            const prompt =
              options.textPrompt ?? sessionOptions.textPrompt?.trim()
            if (!prompt) {
              notify('pages.autoplay.notifications.needText', {
                type: 'warning',
              })
              return null
            }
            return dataProvider.getTextRecommendations({
              ...payloadBase,
              text: prompt,
            })
          }
          case 'custom': {
            const seedId = options.seedSongId || selectedSeed?.id
            if (!seedId) {
              notify('pages.autoplay.notifications.needSeed', {
                type: 'warning',
              })
              return null
            }
            return dataProvider.getCustomRecommendations({
              ...payloadBase,
              songIds: [seedId],
            })
          }
          default:
            return dataProvider.getRecentRecommendations(payloadBase)
        }
      }

      try {
        const request = await determineRequest()
        if (!request) {
          return
        }
        const { data } = request
        if (data?.resultSource !== 'semantic' || data?.degraded === true) {
          notify(
            translate('pages.autoplay.notifications.semanticUnavailable', {
              _: 'Semantic recommendations are unavailable right now.',
            }),
            { type: 'warning' },
          )
          return
        }
        const tracks = data?.tracks || []
        if (!tracks.length) {
          notify('pages.autoplay.notifications.noNew', { type: 'warning' })
          return
        }
        const trackMap = {}
        const newIds = []
        const seedIds =
          mode === 'custom' && selectedSeed?.id
            ? new Set([selectedSeed.id])
            : null
        tracks.forEach((track) => {
          if (!track || !track.id) {
            return
          }
          if (seedIds && seedIds.has(track.id)) {
            return
          }
          if (
            autoplay?.requestedTrackIds?.includes(track.id) ||
            autoplay?.playedTrackIds?.includes(track.id)
          ) {
            return
          }
          trackMap[track.id] = track
          newIds.push(track.id)
        })
        if (!newIds.length) {
          notify('pages.autoplay.notifications.noNew', { type: 'warning' })
          return
        }
        if (player.queue.length === 0) {
          dispatch(playTracks(trackMap, newIds, newIds[0]))
        } else {
          dispatch(addTracks(trackMap, newIds))
        }
        dispatch(markAutoPlayTracksRequested(newIds, `start:${mode}`))
        if (Array.isArray(data?.warnings) && data.warnings.length > 0) {
          data.warnings.forEach((warning) =>
            notify(sanitizeRecommendationWarning(warning, translate), {
              type: 'info',
            }),
          )
        }
        dispatch(setAutoPlayEnabled(true))
        dataProvider
          .updateAutoPlaySettings({
            ...normalizeAutoPlaySettings(sessionOptions),
            enabled: true,
          })
          .then(({ data: updated }) => dispatch(syncAutoPlaySettings(updated)))
          .catch(() => {
            notify('pages.autoplay.settings.serverError', { type: 'warning' })
          })
      } catch (error) {
        notify(
          formatRecommendationError(
            error,
            translate,
            translate('ra.page.error', { _: 'Unable to load the next songs.' }),
          ),
          { type: 'warning' },
        )
      } finally {
        fetchingRef.current = false
        setFetching(false)
      }
    },
    [
      sessionOptions,
      dataProvider,
      notify,
      selectedSeed,
      ensureUniquePositive,
      buildExcludeIds,
      player.queue.length,
      dispatch,
      recommendationHealth,
      translate,
      autoplay,
    ],
  )

  const handleStartSession = () => {
    resetFeedback()
    fetchRecommendations({
      mode: sessionOptions.mode,
      seedSongId: selectedSeed?.id,
    })
  }

  const handleFetchMore = () => {
    refillAutoPlayQueue({
      autoplay,
      dataProvider,
      dispatch,
      notify,
      player,
      silent: false,
      source: 'page',
      translate,
    })
  }

  const handleClearQueue = () => {
    dispatch(clearQueue())
  }

  const modeOptions = useMemo(
    () =>
      AUTO_MODE_OPTIONS(translate).map((option) => ({
        ...option,
        disabled: !isRecommendationModeAvailable(
          recommendationHealth,
          option.value,
        ),
      })),
    [translate, recommendationHealth],
  )
  const modeLookup = useMemo(() => {
    const lookup = {}
    modeOptions.forEach((option) => {
      lookup[option.value] = option.label
    })
    return lookup
  }, [modeOptions])
  const currentModeLabel =
    modeLookup[sessionOptions.mode] || sessionOptions.mode
  const currentModeMessage = healthMessageForMode(
    recommendationHealth,
    sessionOptions.mode,
    translate,
  )

  const toggleFeedback = useCallback(
    (trackId, direction) => {
      dispatch(toggleAutoPlayFeedback(trackId, direction))
    },
    [dispatch],
  )

  return (
    <Box className={classes.root}>
      <Title title={translate('pages.autoplay.title', { _: 'Auto Play' })} />
      <Tabs
        value={tab}
        onChange={handleTabChange}
        indicatorColor="primary"
        textColor="primary"
        className={classes.tabs}
      >
        <Tab
          label={translate('pages.autoplay.mixTab', { _: 'Mix' })}
          value={0}
        />
        <Tab
          label={translate('pages.autoplay.settingsTab', { _: 'Settings' })}
          value={1}
        />
      </Tabs>

      {tab === 0 && (
        <>
          {!recommendationHealthLoading &&
            recommendationHealth.status !== 'ready' && (
              <Card className={classes.statusBanner} variant="outlined">
                <Typography variant="subtitle2">
                  {recommendationHealth.engine?.ready
                    ? translate('pages.autoplay.health.textUnavailable', {
                        _: 'Text recommendations unavailable',
                      })
                    : translate('pages.autoplay.health.engineUnavailable', {
                        _: 'Semantic recommendations unavailable',
                      })}
                </Typography>
                <Typography variant="body2" className={classes.exclusionHint}>
                  {currentModeMessage}
                </Typography>
              </Card>
            )}
          <Card className={classes.controlsCard} variant="outlined">
            <Typography variant="h5">
              {translate('pages.autoplay.controls.title', {
                _: 'Build your next queue',
              })}
            </Typography>
            <Box className={classes.controlGrid}>
              <Box className={classes.modeSummary}>
                <Typography
                  variant="subtitle1"
                  className={classes.modeSummaryLine}
                >
                  {translate('pages.autoplay.controls.modeSummary', {
                    _: 'Listening mode: %{mode}',
                    mode: currentModeLabel,
                  })}
                </Typography>
                <Typography variant="body2" className={classes.exclusionHint}>
                  {translate('pages.autoplay.controls.modeSettingsHint', {
                    _: 'Change your listening mode & more in the Settings tab.',
                  })}
                </Typography>
                {!isRecommendationModeAvailable(
                  recommendationHealth,
                  sessionOptions.mode,
                ) && (
                  <Typography variant="body2" className={classes.warning}>
                    {currentModeMessage}
                  </Typography>
                )}
              </Box>

              {(sessionOptions.mode === 'text' ||
                sessionOptions.mode === 'custom') && (
                <TextField
                  variant="outlined"
                  className={classes.fullWidthGridItem}
                  label={
                    sessionOptions.mode === 'text'
                      ? translate('pages.autoplay.controls.textPromptLabel', {
                          _: 'Text prompt',
                        })
                      : translate('pages.autoplay.controls.seedSearchLabel', {
                          _: 'Seed song',
                        })
                  }
                  multiline={sessionOptions.mode === 'text'}
                  minRows={sessionOptions.mode === 'text' ? 3 : 1}
                  value={
                    sessionOptions.mode === 'text'
                      ? sessionOptions.textPrompt
                      : seedQuery
                  }
                  onChange={(event) => {
                    if (sessionOptions.mode === 'text') {
                      updateSessionOption('textPrompt', event.target.value)
                    } else {
                      setSeedQuery(event.target.value)
                    }
                  }}
                  placeholder={
                    sessionOptions.mode === 'text'
                      ? translate(
                          'pages.autoplay.controls.textPromptPlaceholder',
                          {
                            _: 'Describe the vibe you want to hear',
                          },
                        )
                      : translate(
                          'pages.autoplay.controls.seedSearchPlaceholder',
                          {
                            _: 'Type to search songs',
                          },
                        )
                  }
                  InputProps={
                    sessionOptions.mode === 'custom'
                      ? {
                          endAdornment: (
                            <InputAdornment position="end">
                              {seedLoading ? (
                                <CircularProgress size={16} />
                              ) : (
                                <SearchIcon fontSize="small" />
                              )}
                            </InputAdornment>
                          ),
                        }
                      : undefined
                  }
                  fullWidth
                  disabled={
                    sessionOptions.mode === 'text' &&
                    !isRecommendationModeAvailable(recommendationHealth, 'text')
                  }
                />
              )}
            </Box>

            {sessionOptions.mode === 'custom' && seedResults.length > 0 && (
              <List className={classes.seedSearchResults} dense>
                {seedResults.map((song) => (
                  <ListItem
                    button
                    key={song.id}
                    selected={selectedSeed?.id === song.id}
                    onClick={() => {
                      setSelectedSeed(song)
                      setSeedResults([])
                      setSeedQuery(song.title)
                    }}
                  >
                    <ListItemText
                      primary={song.title}
                      secondary={`${song.artist} · ${song.album}`}
                    />
                  </ListItem>
                ))}
              </List>
            )}

            {selectedSeed && sessionOptions.mode === 'custom' && (
              <Box className={classes.chipRow}>
                <Chip
                  label={`${selectedSeed.title} · ${selectedSeed.artist}`}
                  onDelete={() => {
                    setSelectedSeed(null)
                    setSeedQuery('')
                  }}
                  deleteIcon={<ClearIcon />}
                />
              </Box>
            )}

            <Box className={classes.actionRow}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<PlayArrowIcon />}
                onClick={handleStartSession}
                disabled={
                  fetching ||
                  !isRecommendationModeAvailable(
                    recommendationHealth,
                    sessionOptions.mode,
                  )
                }
                size="large"
                className={classes.primaryActionButton}
              >
                {fetching
                  ? translate('pages.autoplay.controls.fetching', {
                      _: 'Building…',
                    })
                  : translate('pages.autoplay.controls.start', {
                      _: 'Start Auto Play',
                    })}
              </Button>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={handleFetchMore}
                disabled={
                  fetching ||
                  !isRecommendationModeAvailable(
                    recommendationHealth,
                    sessionOptions.mode,
                  )
                }
              >
                {translate('pages.autoplay.controls.more', { _: 'Add more' })}
              </Button>
              <Button variant="text" onClick={resetFeedback}>
                {translate('pages.autoplay.controls.resetFeedback', {
                  _: 'Reset feedback',
                })}
              </Button>
              <Button variant="text" onClick={handleClearQueue}>
                {translate('pages.autoplay.controls.clearQueue', {
                  _: 'Clear queue',
                })}
              </Button>
            </Box>
          </Card>

          <Card className={classes.queueCard} variant="outlined">
            <Typography variant="h6" gutterBottom>
              {translate('pages.autoplay.queue.title', { _: 'Upcoming queue' })}
            </Typography>
            {player.queue.length === 0 ? (
              <Typography variant="body2" color="textSecondary">
                {translate('pages.autoplay.queue.empty', {
                  _: 'Start Auto Play to fill your listening queue.',
                })}
              </Typography>
            ) : (
              <List className={classes.queueList} dense>
                {player.queue.map((item) => {
                  const resolvedId =
                    item.trackId ||
                    item.song?.id ||
                    item.song?.mediaFileId ||
                    ''
                  const normalizedId =
                    typeof resolvedId === 'string'
                      ? resolvedId.trim()
                      : resolvedId.toString().trim()
                  const isPositive =
                    normalizedId !== '' &&
                    autoplay?.positiveTrackIds?.includes(normalizedId)
                  const isNegative =
                    normalizedId !== '' &&
                    autoplay?.negativeTrackIds?.includes(normalizedId)
                  const disabledFeedback = normalizedId === ''

                  return (
                    <ListItem
                      key={item.uuid}
                      selected={item.uuid === player.current?.uuid}
                    >
                      <ListItemText
                        primary={item.song?.title || item.name}
                        secondary={`${item.song?.artist || ''} · ${item.song?.album || ''}`}
                      />
                      <ListItemSecondaryAction>
                        <Tooltip
                          title={translate('pages.autoplay.feedback.like', {
                            _: 'Thumbs up',
                          })}
                        >
                          <span>
                            <IconButton
                              size="small"
                              color={isPositive ? 'primary' : 'default'}
                              onClick={() => toggleFeedback(normalizedId, 'up')}
                              disabled={disabledFeedback}
                              aria-label={translate(
                                'pages.autoplay.feedback.like',
                                { _: 'Thumbs up' },
                              )}
                            >
                              <ThumbUpAltOutlinedIcon fontSize="small" />
                            </IconButton>
                          </span>
                        </Tooltip>
                        <Tooltip
                          title={translate('pages.autoplay.feedback.dislike', {
                            _: 'Thumbs down',
                          })}
                        >
                          <span>
                            <IconButton
                              size="small"
                              color={isNegative ? 'secondary' : 'default'}
                              onClick={() =>
                                toggleFeedback(normalizedId, 'down')
                              }
                              disabled={disabledFeedback}
                              aria-label={translate(
                                'pages.autoplay.feedback.dislike',
                                { _: 'Thumbs down' },
                              )}
                            >
                              <ThumbDownAltOutlinedIcon fontSize="small" />
                            </IconButton>
                          </span>
                        </Tooltip>
                      </ListItemSecondaryAction>
                    </ListItem>
                  )
                })}
              </List>
            )}
          </Card>
        </>
      )}

      {tab === 1 && (
        <AutoPlaySettingsPanel
          translate={translate}
          draft={settingsDraft}
          onFieldChange={updateSettingsDraft}
          onSave={handleSaveSettings}
          onReset={() => setSettingsDraft(savedSettings)}
          saving={savingSettings}
          loading={settingsLoading || playlistsLoading}
          error={settingsError}
          playlists={playlists}
          modeOptions={modeOptions}
        />
      )}
    </Box>
  )
}

export default AutoPlayPage
