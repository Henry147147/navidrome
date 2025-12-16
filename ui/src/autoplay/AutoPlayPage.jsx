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
import { addTracks, playTracks, clearQueue } from '../actions'
import { computeFeedback } from './feedbackUtils'

const DEFAULT_SETTINGS = {
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
}))

const AutoPlayPage = () => {
  const classes = useStyles()
  const translate = useTranslate()
  const notify = useNotify()
  const dataProvider = useDataProvider()
  const dispatch = useDispatch()
  const player = useSelector((state) => state.player)

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
  const [sessionActive, setSessionActive] = useState(false)
  const [positiveTrackIds, setPositiveTrackIds] = useState([])
  const [negativeTrackIds, setNegativeTrackIds] = useState([])

  const playedIdsRef = useRef(new Set())
  const requestedIdsRef = useRef(new Set())
  const positiveRef = useRef([])
  const negativeRef = useRef([])

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
        const normalized = {
          mode: data?.mode || DEFAULT_SETTINGS.mode,
          textPrompt: data?.textPrompt || '',
          excludePlaylistIds: data?.excludePlaylistIds || [],
          batchSize: DEFAULT_SETTINGS.batchSize,
          diversityOverride:
            data?.diversityOverride === undefined
              ? null
              : data?.diversityOverride,
        }
        setSettingsDraft(normalized)
        setSessionOptions(normalized)
        setSavedSettings(normalized)
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
  }, [dataProvider])

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
    setPositiveTrackIds([])
    setNegativeTrackIds([])
    playedIdsRef.current = new Set()
    requestedIdsRef.current = new Set()
  }

  useEffect(() => {
    positiveRef.current = positiveTrackIds
  }, [positiveTrackIds])

  useEffect(() => {
    negativeRef.current = negativeTrackIds
  }, [negativeTrackIds])

  const handleSaveSettings = () => {
    setSavingSettings(true)
    const payload = {
      ...settingsDraft,
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
        const normalized = {
          mode: data?.mode || DEFAULT_SETTINGS.mode,
          textPrompt: data?.textPrompt || '',
          excludePlaylistIds: data?.excludePlaylistIds || [],
          batchSize: DEFAULT_SETTINGS.batchSize,
          diversityOverride:
            data?.diversityOverride === undefined
              ? null
              : data?.diversityOverride,
        }
        setSettingsDraft(normalized)
        setSessionOptions((prev) => ({ ...prev, ...normalized }))
        setSavedSettings(normalized)
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

  const resolveTextSeeds = useCallback(
    async (prompt) => {
      const query = prompt?.trim()
      if (!query) {
        return []
      }
      const { data } = await dataProvider.getList('song', {
        pagination: { page: 1, perPage: 25 },
        sort: { field: 'playDate', order: 'DESC' },
        filter: { q: query },
      })
      return (data || [])
        .map((song) => song.id)
        .filter(Boolean)
        .slice(0, 10)
    },
    [dataProvider],
  )

  useEffect(() => {
    if (sessionOptions.mode !== 'custom') {
      setSelectedSeed(null)
      setSeedResults([])
      setSeedQuery('')
    }
  }, [sessionOptions.mode])

  const buildExcludeIds = useCallback(() => {
    const combined = new Set([
      ...Array.from(playedIdsRef.current),
      ...negativeTrackIds,
    ])
    return Array.from(combined)
  }, [negativeTrackIds])

  const ensureUniquePositive = useCallback(() => {
    return Array.from(new Set(positiveTrackIds))
  }, [positiveTrackIds])

  const fetchRecommendations = useCallback(
    async (options = {}) => {
      if (fetchingRef.current) {
        return
      }
      fetchingRef.current = true
      setFetching(true)

      const mode = options.mode || sessionOptions.mode || DEFAULT_SETTINGS.mode
      const excludePlaylistIds =
        options.excludePlaylistIds || sessionOptions.excludePlaylistIds || []
      const excludeTrackIds = buildExcludeIds()
      const positiveIds = ensureUniquePositive()

      const payloadBase = {
        limit: sessionOptions.batchSize || DEFAULT_SETTINGS.batchSize,
        excludeTrackIds,
        excludePlaylistIds,
        positiveTrackIds: positiveIds,
        negativeTrackIds,
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
            const songIds = await resolveTextSeeds(prompt)
            if (songIds.length === 0) {
              notify('pages.autoplay.notifications.noSeeds', {
                type: 'warning',
              })
              return null
            }
            return dataProvider.getCustomRecommendations({
              ...payloadBase,
              songIds,
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
        const tracks = data?.tracks || []
        if (!tracks.length) {
          notify('pages.autoplay.notifications.noNew', { type: 'warning' })
          return
        }
        const trackMap = {}
        const newIds = []
        tracks.forEach((track) => {
          if (!track || !track.id) {
            return
          }
          if (
            requestedIdsRef.current.has(track.id) ||
            playedIdsRef.current.has(track.id)
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
        newIds.forEach((id) => requestedIdsRef.current.add(id))
        if (player.queue.length === 0) {
          dispatch(playTracks(trackMap, newIds, newIds[0]))
        } else {
          dispatch(addTracks(trackMap, newIds))
        }
        if (Array.isArray(data?.warnings) && data.warnings.length > 0) {
          data.warnings.forEach((warning) => notify(warning, { type: 'info' }))
        }
        setSessionActive(true)
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Auto Play fetch failed', error)
        notify('ra.page.error', { type: 'warning' })
      } finally {
        fetchingRef.current = false
        setFetching(false)
      }
    },
    [
      sessionOptions,
      dataProvider,
      notify,
      resolveTextSeeds,
      selectedSeed,
      negativeTrackIds,
      ensureUniquePositive,
      buildExcludeIds,
      player.queue.length,
      dispatch,
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
    fetchRecommendations({ mode: sessionOptions.mode })
  }

  const handleClearQueue = () => {
    dispatch(clearQueue())
    resetFeedback()
    setSessionActive(false)
  }

  const remainingQueue = useMemo(() => {
    if (player.queue.length === 0) {
      return 0
    }
    const currentUuid = player.current?.uuid
    const currentIndex = player.queue.findIndex(
      (item) => item.uuid === currentUuid,
    )
    if (currentIndex === -1) {
      return player.queue.length
    }
    return player.queue.length - currentIndex - 1
  }, [player])

  useEffect(() => {
    const trackId = player.current?.trackId || player.current?.song?.id
    if (trackId) {
      playedIdsRef.current.add(trackId)
    }
  }, [player])

  useEffect(() => {
    if (!sessionActive) {
      return
    }
    const bufferThreshold = Math.max(
      3,
      Math.floor((sessionOptions.batchSize || DEFAULT_SETTINGS.batchSize) / 2),
    )
    if (remainingQueue <= bufferThreshold && !fetchingRef.current) {
      fetchRecommendations({ mode: sessionOptions.mode })
    }
  }, [sessionActive, remainingQueue, fetchRecommendations, sessionOptions])

  const modeOptions = useMemo(() => AUTO_MODE_OPTIONS(translate), [translate])
  const modeLookup = useMemo(() => {
    const lookup = {}
    modeOptions.forEach((option) => {
      lookup[option.value] = option.label
    })
    return lookup
  }, [modeOptions])
  const currentModeLabel =
    modeLookup[sessionOptions.mode] || sessionOptions.mode

  const toggleFeedback = useCallback((trackId, direction) => {
    const { positive, negative } = computeFeedback(
      positiveRef.current,
      negativeRef.current,
      trackId,
      direction,
    )
    setPositiveTrackIds(positive)
    setNegativeTrackIds(negative)
    positiveRef.current = positive
    negativeRef.current = negative
  }, [])

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
                disabled={fetching}
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
                disabled={fetching}
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
                    positiveTrackIds.includes(normalizedId)
                  const isNegative =
                    normalizedId !== '' &&
                    negativeTrackIds.includes(normalizedId)
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
