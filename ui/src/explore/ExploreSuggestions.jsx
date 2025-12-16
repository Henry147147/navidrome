import React, { useEffect, useMemo, useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardActionArea,
  CardContent,
  CircularProgress,
  Divider,
  IconButton,
  InputAdornment,
  List,
  ListItem,
  ListItemText,
  Paper,
  TextField,
  Tooltip,
  Typography,
  Chip,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Checkbox,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import {
  Title,
  useDataProvider,
  useNotify,
  useQueryWithStore,
  useRefresh,
  useTranslate,
} from 'react-admin'
import { Link } from 'react-router-dom'
import ExploreIcon from '@material-ui/icons/Explore'
import QueueMusicIcon from '@material-ui/icons/QueueMusic'
import PlaylistAddIcon from '@material-ui/icons/PlaylistAdd'
import SearchIcon from '@material-ui/icons/Search'
import CloseIcon from '@material-ui/icons/Close'
import AutorenewIcon from '@material-ui/icons/Autorenew'
import config from '../config'
import { BRAND_NAME } from '../consts'
import { formatDuration } from '../utils'
import ExploreSettingsPanel from './ExploreSettingsPanel'
import TextPlaylistGenerator from './TextPlaylistGenerator'

const useStyles = makeStyles((theme) => ({
  page: {
    marginTop: theme.spacing(2),
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(4),
  },
  tabs: {
    alignSelf: 'flex-start',
  },
  tabPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(4),
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(2),
  },
  heroCard: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: theme.spacing(2),
    padding: theme.spacing(3),
  },
  heroIcon: {
    color: theme.palette.primary.main,
    fontSize: 48,
  },
  recommendationCard: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(2),
    padding: theme.spacing(3),
  },
  recommendationList: {
    maxHeight: 280,
    overflowY: 'auto',
    borderRadius: theme.shape.borderRadius,
    border: `1px solid ${theme.palette.divider}`,
  },
  recommendationHeader: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing(1),
  },
  playlistHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
    gap: theme.spacing(2),
  },
  card: {
    height: '100%',
  },
  cardAction: {
    height: '100%',
    alignItems: 'stretch',
  },
  cardMeta: {
    color: theme.palette.text.secondary,
    marginTop: theme.spacing(1),
  },
  placeholder: {
    color: theme.palette.text.secondary,
  },
  error: {
    color: theme.palette.error.main,
  },
  warning: {
    color: theme.palette.warning.main,
    marginTop: theme.spacing(1),
  },
  buttonRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(1),
    alignItems: 'center',
  },
  searchResults: {
    marginTop: theme.spacing(1),
    maxHeight: 260,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
  },
  selectedSongsContainer: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(1),
  },
  playlistExclusionControl: {
    marginTop: theme.spacing(2),
    minWidth: 240,
  },
  selectChips: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(0.5),
  },
  selectChip: {
    margin: 0,
  },
  listItemPrimary: {
    fontWeight: theme.typography.fontWeightMedium,
  },
  trackActions: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
    marginLeft: 'auto',
    paddingLeft: theme.spacing(1),
  },
  trackActionButton: {
    padding: theme.spacing(0.5),
  },
  loadMoreContainer: {
    padding: theme.spacing(1),
    borderTop: `1px solid ${theme.palette.divider}`,
    display: 'flex',
    justifyContent: 'center',
  },
}))

const PLAYLIST_LIMIT = Math.min(6, config.maxSidebarPlaylists || 6)
const DEFAULT_SETTINGS = {
  mixLength: 25,
  baseDiversity: 0.15,
  discoveryExploration: 0.6,
  seedRecencyWindowDays: 60,
  favoritesBlendWeight: 0.85,
  lowRatingPenalty: 0.85,
}

const RecommendationPreview = ({
  result,
  playlistName,
  onPlaylistNameChange,
  onSave,
  saving,
  translate,
  allowTrackActions = false,
  onRemoveTrack,
  onRerollTrack,
  pendingTrackId,
}) => {
  const classes = useStyles()

  const tracks = (result && result.tracks) || []
  const trackIds = (result && result.trackIds) || []

  if (!result) {
    return null
  }

  return (
    <Box>
      <Box className={classes.recommendationHeader}>
        <Typography variant="h6">
          {translate('pages.explore.recommendationTitle', {
            _: 'Preview: %{name}',
            name: playlistName || result.name,
          })}
        </Typography>
        <TextField
          variant="outlined"
          label={translate('pages.explore.playlistNameLabel', {
            _: 'Playlist name',
          })}
          value={playlistName}
          onChange={(event) => onPlaylistNameChange(event.target.value)}
          fullWidth
        />
      </Box>

      <Paper className={classes.recommendationList} variant="outlined">
        <List dense disablePadding>
          {tracks.map((track, index) => {
            const trackId =
              track.id || track.ID || trackIds[index] || `${index}`
            const isUpdating =
              Boolean(pendingTrackId) && pendingTrackId === trackId
            return (
              <React.Fragment key={trackId}>
                <ListItem alignItems="flex-start">
                  <ListItemText
                    primary={
                      <span className={classes.listItemPrimary}>
                        {track.title}
                      </span>
                    }
                    secondary={`${track.artist || ''} • ${track.album || ''}`}
                  />
                  <Box className={classes.trackActions}>
                    {track.duration && (
                      <Typography variant="caption">
                        {formatDuration(track.duration)}
                      </Typography>
                    )}
                    {allowTrackActions && (
                      <>
                        {onRerollTrack && (
                          <Tooltip
                            title={translate('pages.explore.rerollTrack', {
                              _: 'Reroll song',
                            })}
                          >
                            <span>
                              <IconButton
                                className={classes.trackActionButton}
                                size="small"
                                onClick={() => onRerollTrack(track, index)}
                                disabled={isUpdating}
                              >
                                {isUpdating ? (
                                  <CircularProgress size={16} />
                                ) : (
                                  <AutorenewIcon fontSize="small" />
                                )}
                              </IconButton>
                            </span>
                          </Tooltip>
                        )}
                        {onRemoveTrack && (
                          <Tooltip
                            title={translate('pages.explore.removeTrack', {
                              _: 'Remove song',
                            })}
                          >
                            <span>
                              <IconButton
                                className={classes.trackActionButton}
                                size="small"
                                onClick={() => onRemoveTrack(track, index)}
                                disabled={isUpdating}
                              >
                                <CloseIcon fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                        )}
                      </>
                    )}
                  </Box>
                </ListItem>
                {index < tracks.length - 1 && <Divider component="li" />}
              </React.Fragment>
            )
          })}
        </List>
      </Paper>

      {result.warnings && result.warnings.length > 0 && (
        <Box className={classes.warning}>
          {result.warnings.map((warning, idx) => (
            <Typography key={idx} variant="body2">
              {translate('pages.explore.recommendationWarning', {
                _: 'Note: %{warning}',
                warning,
              })}
            </Typography>
          ))}
        </Box>
      )}

      <Box className={classes.buttonRow}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<PlaylistAddIcon />}
          onClick={onSave}
          disabled={saving || trackIds.length === 0}
        >
          {saving ? (
            <CircularProgress size={18} color="inherit" />
          ) : (
            translate('pages.explore.savePlaylist', { _: 'Save as playlist' })
          )}
        </Button>
        <Typography variant="body2" className={classes.placeholder}>
          {translate('pages.explore.trackCountLabel', {
            _: '%{count} tracks',
            count: trackIds.length,
          })}
        </Typography>
      </Box>
    </Box>
  )
}

const ExploreSuggestions = () => {
  const classes = useStyles()
  const translate = useTranslate()
  const dataProvider = useDataProvider()
  const notify = useNotify()
  const refresh = useRefresh()
  const [activeTab, setActiveTab] = useState(0)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [settingsDraft, setSettingsDraft] = useState(DEFAULT_SETTINGS)
  const [settingsErrors, setSettingsErrors] = useState({})
  const [settingsDirty, setSettingsDirty] = useState(false)
  const [settingsLoading, setSettingsLoading] = useState(false)
  const [settingsSaving, setSettingsSaving] = useState(false)
  const [settingsMessage, setSettingsMessage] = useState(null)

  // Multi-model recommendation options
  const [selectedModels, setSelectedModels] = useState(['muq'])
  const [mergeStrategy, setMergeStrategy] = useState('union')
  const [minModelAgreement, setMinModelAgreement] = useState(1)

  const {
    data: playlistMap,
    loading: playlistsLoading,
    loaded: playlistsLoaded,
    error: playlistsError,
  } = useQueryWithStore({
    type: 'getList',
    resource: 'playlist',
    payload: {
      pagination: {
        page: 1,
        perPage: PLAYLIST_LIMIT,
      },
      sort: { field: 'updatedAt', order: 'DESC' },
    },
  })

  const playlists = useMemo(() => {
    if (!playlistMap) {
      return []
    }
    return Object.keys(playlistMap).map((key) => playlistMap[key])
  }, [playlistMap])

  useEffect(() => {
    let isActive = true
    setSettingsLoading(true)
    dataProvider
      .getRecommendationSettings()
      .then(({ data }) => {
        if (!isActive) {
          return
        }
        const merged = { ...DEFAULT_SETTINGS, ...data }
        setSettings(merged)
        setSettingsDraft(merged)
        setSettingsDirty(false)
        setSettingsErrors({})
        setSettingsMessage(null)
      })
      .catch((error) => {
        if (!isActive) {
          return
        }
        const message =
          error?.body?.message ||
          error?.message ||
          translate('pages.explore.settings.loadError', {
            _: 'Unable to load settings.',
          })
        setSettingsMessage(message)
      })
      .finally(() => {
        if (isActive) {
          setSettingsLoading(false)
        }
      })
    return () => {
      isActive = false
    }
  }, [dataProvider, translate])

  const createGeneratorState = () => ({
    result: null,
    name: '',
    loading: false,
    error: null,
    exclude: [],
    updatingTrackId: null,
  })
  const [generators, setGenerators] = useState(() => ({
    recent: createGeneratorState(),
    favorites: createGeneratorState(),
    all: createGeneratorState(),
    discovery: createGeneratorState(),
  }))
  const generatorDefaultFallback = {
    recent: 'Recent Mix',
    favorites: 'Favorites Mix',
    all: 'All Metrics Mix',
    discovery: 'Discovery Mix',
  }
  const generatorErrorFallback = {
    recent: 'Play a few songs and try again.',
    favorites: 'Star or rate a few songs and try again.',
    all: "We didn't find enough signals to build this mix yet.",
    discovery: 'We need more listening data before exploring.',
  }
  const generatorApiMap = {
    recent: dataProvider.getRecentRecommendations,
    favorites: dataProvider.getFavoriteRecommendations,
    all: dataProvider.getAllRecommendations,
    discovery: dataProvider.getDiscoveryRecommendations,
  }
  const handleTabChange = (_, value) => {
    setActiveTab(value)
  }
  const handleSettingsFieldChange = (key, value) => {
    setSettingsDraft((prev) => ({
      ...prev,
      [key]: value,
    }))
    setSettingsDirty(true)
    setSettingsErrors((prev) => ({
      ...prev,
      [key]: undefined,
    }))
    setSettingsMessage(null)
  }
  const validateSettings = (values) => {
    const issues = {}
    if (values.mixLength < 10 || values.mixLength > 100) {
      issues.mixLength = translate('pages.explore.settings.mixLengthError', {
        _: 'Choose a value between 10 and 100 tracks.',
      })
    }
    if (values.baseDiversity < 0 || values.baseDiversity > 1) {
      issues.baseDiversity = translate(
        'pages.explore.settings.baseDiversityError',
        { _: 'Set a value between 0% and 100%.' },
      )
    }
    if (values.discoveryExploration < 0.3 || values.discoveryExploration > 1) {
      issues.discoveryExploration = translate(
        'pages.explore.settings.discoveryExplorationError',
        { _: 'Set a value between 30% and 100%.' },
      )
    }
    if (
      values.seedRecencyWindowDays < 7 ||
      values.seedRecencyWindowDays > 120
    ) {
      issues.seedRecencyWindowDays = translate(
        'pages.explore.settings.recencyWindowError',
        { _: 'Use a window between 7 and 120 days.' },
      )
    }
    if (values.favoritesBlendWeight < 0.1 || values.favoritesBlendWeight > 1) {
      issues.favoritesBlendWeight = translate(
        'pages.explore.settings.favoritesWeightError',
        { _: 'Keep the weight between 10% and 100%.' },
      )
    }
    if (values.lowRatingPenalty < 0.3 || values.lowRatingPenalty > 1) {
      issues.lowRatingPenalty = translate(
        'pages.explore.settings.lowRatingPenaltyError',
        { _: 'Keep the penalty between 30% and 100%.' },
      )
    }
    return issues
  }
  const handleSettingsSave = () => {
    const validation = validateSettings(settingsDraft)
    setSettingsErrors(validation)
    if (Object.keys(validation).length > 0) {
      return
    }
    setSettingsSaving(true)
    setSettingsMessage(null)
    dataProvider
      .updateRecommendationSettings(settingsDraft)
      .then(({ data }) => {
        const merged = { ...DEFAULT_SETTINGS, ...data }
        setSettings(merged)
        setSettingsDraft(merged)
        setSettingsDirty(false)
        setSettingsErrors({})
        setSettingsMessage(null)
        notify(
          translate('pages.explore.settings.saveSuccess', {
            _: 'Recommendation settings updated.',
          }),
          'info',
        )
      })
      .catch((error) => {
        const message =
          error?.body?.message ||
          error?.message ||
          translate('pages.explore.settings.saveError', {
            _: 'Unable to save settings.',
          })
        setSettingsMessage(message)
      })
      .finally(() => {
        setSettingsSaving(false)
      })
  }
  const handleSettingsReset = () => {
    setSettingsDraft(settings)
    setSettingsDirty(false)
    setSettingsErrors({})
    setSettingsMessage(null)
  }
  const updateGeneratorName = (mode, value) => {
    setGenerators((prev) => ({
      ...prev,
      [mode]: { ...prev[mode], name: value },
    }))
  }
  const runGenerator = (
    mode,
    requestOptions = {},
    defaultNameKey,
    errorKey,
  ) => {
    const api = generatorApiMap[mode]
    if (!api) {
      return
    }
    const payload = {
      limit: settings.mixLength,
      diversity: settings.baseDiversity,
      // Multi-model options
      models: selectedModels.length > 0 ? selectedModels : ['muq'],
      mergeStrategy: selectedModels.length > 1 ? mergeStrategy : undefined,
      minModelAgreement:
        selectedModels.length > 1 ? minModelAgreement : undefined,
      ...requestOptions,
    }
    setGenerators((prev) => ({
      ...prev,
      [mode]: {
        ...prev[mode],
        loading: true,
        error: null,
        updatingTrackId: null,
      },
    }))
    api(payload)
      .then(({ data }) => {
        const autoName =
          data?.name ||
          translate(defaultNameKey, {
            _: generatorDefaultFallback[mode],
          })
        setGenerators((prev) => ({
          ...prev,
          [mode]: {
            ...prev[mode],
            loading: false,
            result: data,
            name:
              prev[mode].name && prev[mode].name.trim()
                ? prev[mode].name
                : autoName,
            error: null,
            exclude: [],
            updatingTrackId: null,
          },
        }))
      })
      .catch((error) => {
        const fallbackMessage = translate(errorKey, {
          _: generatorErrorFallback[mode],
        })
        const message =
          error?.body?.message || error?.message || fallbackMessage
        setGenerators((prev) => ({
          ...prev,
          [mode]: {
            ...prev[mode],
            loading: false,
            error: message,
            result: null,
            exclude: [],
            updatingTrackId: null,
          },
        }))
      })
  }
  const [saving, setSaving] = useState(false)
  const generatorConfigs = [
    {
      key: 'recent',
      title: translate('pages.explore.recentTitle', {
        _: 'Generate from recent listens',
      }),
      description: translate('pages.explore.recentDescription', {
        _: 'Build a personalised mix using the tracks you have been enjoying lately.',
      }),
      defaultNameKey: 'pages.explore.recentDefaultName',
      errorKey: 'pages.explore.noRecentSeeds',
      onGenerate: () =>
        runGenerator(
          'recent',
          {},
          'pages.explore.recentDefaultName',
          'pages.explore.noRecentSeeds',
        ),
    },
    {
      key: 'favorites',
      title: translate('pages.explore.favoritesTitle', {
        _: 'Generate from liked songs',
      }),
      description: translate('pages.explore.favoritesDescription', {
        _: 'Use the songs you have starred or rated highly to create a mix.',
      }),
      defaultNameKey: 'pages.explore.favoritesDefaultName',
      errorKey: 'pages.explore.noFavoritesSeeds',
      onGenerate: () =>
        runGenerator(
          'favorites',
          {},
          'pages.explore.favoritesDefaultName',
          'pages.explore.noFavoritesSeeds',
        ),
    },
    {
      key: 'all',
      title: translate('pages.explore.allTitle', {
        _: 'Generate from all metrics',
      }),
      description: translate('pages.explore.allDescription', {
        _: 'Blend your recent plays and favourites for a balanced playlist.',
      }),
      defaultNameKey: 'pages.explore.allDefaultName',
      errorKey: 'pages.explore.noAllSeeds',
      onGenerate: () =>
        runGenerator(
          'all',
          {},
          'pages.explore.allDefaultName',
          'pages.explore.noAllSeeds',
        ),
    },
    {
      key: 'discovery',
      title: translate('pages.explore.discoveryTitle', {
        _: 'Generate discovery playlist',
      }),
      description: translate('pages.explore.discoveryDescription', {
        _: 'Surface songs that are farther from your usual listening. Tune exploration from the Settings tab.',
      }),
      defaultNameKey: 'pages.explore.discoveryDefaultName',
      errorKey: 'pages.explore.noDiscoverySeeds',
      onGenerate: () =>
        runGenerator(
          'discovery',
          { diversity: settings.discoveryExploration },
          'pages.explore.discoveryDefaultName',
          'pages.explore.noDiscoverySeeds',
        ),
    },
  ]
  const renderGeneratorCard = (config) => {
    const state = generators[config.key] || {
      result: null,
      name: '',
      loading: false,
      error: null,
    }
    return (
      <Card
        key={config.key}
        className={classes.recommendationCard}
        variant="outlined"
      >
        <Typography variant="h5">{config.title}</Typography>
        {config.description && (
          <Typography variant="body2" className={classes.placeholder}>
            {config.description}
          </Typography>
        )}
        {config.key === 'discovery' && (
          <Typography variant="body2" className={classes.placeholder}>
            {translate('pages.explore.discoverySliderValue', {
              _: '%{value}% exploratory',
              value: Math.round(settings.discoveryExploration * 100),
            })}
            {` · ${translate('pages.explore.settings.adjustInSettings', {
              _: 'Adjust this in the Settings tab.',
            })}`}
          </Typography>
        )}
        <Box className={classes.buttonRow}>
          <Button
            variant="contained"
            color="primary"
            onClick={config.onGenerate}
            disabled={state.loading}
          >
            {state.loading ? (
              <CircularProgress size={18} color="inherit" />
            ) : (
              translate('pages.explore.generateButton', {
                _: 'Generate mix',
              })
            )}
          </Button>
          {state.error && (
            <Typography variant="body2" className={classes.error}>
              {state.error}
            </Typography>
          )}
        </Box>
        <RecommendationPreview
          result={state.result}
          playlistName={state.name}
          onPlaylistNameChange={(value) =>
            updateGeneratorName(config.key, value)
          }
          onSave={() => saveAsPlaylist(state.result, state.name)}
          saving={saving}
          translate={translate}
          allowTrackActions
          onRemoveTrack={(track, index) =>
            handleRemoveGeneratedTrack(config.key, track, index)
          }
          onRerollTrack={(track, index) =>
            handleRerollGeneratedTrack(config.key, track, index)
          }
          pendingTrackId={state.updatingTrackId}
        />
      </Card>
    )
  }

  const [customResult, setCustomResult] = useState(null)
  const [customName, setCustomName] = useState('')
  const [customLoading, setCustomLoading] = useState(false)
  const [customError, setCustomError] = useState(null)
  const [customExcludeIds, setCustomExcludeIds] = useState([])
  const [customUpdatingTrackId, setCustomUpdatingTrackId] = useState(null)

  const SONG_SEARCH_PER_PAGE = 10
  const createInitialSearchState = () => ({
    items: [],
    page: 1,
    total: 0,
    hasMore: false,
    query: '',
  })
  const [songQuery, setSongQuery] = useState('')
  const [searchState, setSearchState] = useState(createInitialSearchState)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchLoadingMore, setSearchLoadingMore] = useState(false)
  const [selectedSongs, setSelectedSongs] = useState([])
  const [excludePlaylistIds, setExcludePlaylistIds] = useState([])

  useEffect(() => {
    const trimmed = songQuery.trim()
    if (!trimmed) {
      setSearchState(createInitialSearchState())
      setSearchLoading(false)
      setSearchLoadingMore(false)
      return undefined
    }

    let isActive = true
    setSearchLoading(true)
    const timer = setTimeout(() => {
      dataProvider
        .getList('song', {
          pagination: { page: 1, perPage: SONG_SEARCH_PER_PAGE },
          sort: { field: 'title', order: 'ASC' },
          filter: { q: trimmed },
        })
        .then((response) => {
          if (!isActive) {
            return
          }
          const items = (response?.data || []).filter(Boolean)
          const total = response?.total ?? items.length
          setSearchState({
            items,
            page: 1,
            total,
            hasMore: SONG_SEARCH_PER_PAGE < total,
            query: trimmed,
          })
        })
        .catch(() => {
          if (isActive) {
            setSearchState({
              items: [],
              page: 1,
              total: 0,
              hasMore: false,
              query: trimmed,
            })
          }
        })
        .finally(() => {
          if (isActive) {
            setSearchLoading(false)
            setSearchLoadingMore(false)
          }
        })
    }, 250)

    return () => {
      isActive = false
      clearTimeout(timer)
    }
  }, [songQuery, dataProvider])

  const handleLoadMoreSearch = () => {
    if (searchLoading || searchLoadingMore || !searchState.hasMore) {
      return
    }
    const query = searchState.query || songQuery.trim()
    if (!query) {
      return
    }
    const nextPage = searchState.page + 1
    setSearchLoading(true)
    setSearchLoadingMore(true)
    dataProvider
      .getList('song', {
        pagination: { page: nextPage, perPage: SONG_SEARCH_PER_PAGE },
        sort: { field: 'title', order: 'ASC' },
        filter: { q: query },
      })
      .then((response) => {
        const items = (response?.data || []).filter(Boolean)
        setSearchState((prev) => {
          const previous = prev || createInitialSearchState()
          const existingItems = previous.items || []
          const existingIds = new Set(
            existingItems
              .map((item) => item && (item.id || item.ID))
              .filter(Boolean),
          )
          const merged = [...existingItems]
          items.forEach((item) => {
            if (!item) {
              return
            }
            const id = item.id || item.ID
            if (id && existingIds.has(id)) {
              return
            }
            if (id) {
              existingIds.add(id)
            }
            merged.push(item)
          })
          const total = response?.total ?? merged.length
          return {
            items: merged,
            page: nextPage,
            total,
            hasMore: nextPage * SONG_SEARCH_PER_PAGE < total,
            query: previous.query || query,
          }
        })
      })
      .catch(() => {
        // keep previous state
      })
      .finally(() => {
        setSearchLoading(false)
        setSearchLoadingMore(false)
      })
  }

  const getSongId = (song) => (song && (song.id || song.ID)) || ''

  const handleSelectSong = (song) => {
    const songId = getSongId(song)
    if (!song || !songId) {
      return
    }
    setSelectedSongs((prev) => {
      if (prev.some((item) => getSongId(item) === songId)) {
        return prev
      }
      return [...prev, song]
    })
    setSongQuery('')
    setSearchState(createInitialSearchState())
    setSearchLoading(false)
    setSearchLoadingMore(false)
  }

  const handleRemoveSong = (id) => {
    setSelectedSongs((prev) => prev.filter((song) => getSongId(song) !== id))
  }

  const handleRemoveGeneratedTrack = (mode, track, index) => {
    setGenerators((prev) => {
      const current = prev[mode]
      if (!current || !current.result) {
        return prev
      }
      const tracks = current.result.tracks || []
      const trackIds = current.result.trackIds || []
      const resolvedIndex =
        typeof index === 'number'
          ? index
          : tracks.findIndex((item) => {
              const candidateId = item?.id || item?.ID
              const targetId = track?.id || track?.ID
              return targetId && candidateId === targetId
            })
      if (resolvedIndex < 0 || resolvedIndex >= tracks.length) {
        return prev
      }
      const removedId =
        trackIds?.[resolvedIndex] || track?.id || track?.ID || ''
      const nextTracks = tracks.filter((_, idx) => idx !== resolvedIndex)
      const nextTrackIds = trackIds.filter((_, idx) => idx !== resolvedIndex)
      const nextExclude = new Set(current.exclude || [])
      if (removedId) {
        nextExclude.add(removedId)
      }
      return {
        ...prev,
        [mode]: {
          ...current,
          result: {
            ...current.result,
            tracks: nextTracks,
            trackIds: nextTrackIds,
          },
          exclude: Array.from(nextExclude),
        },
      }
    })
  }

  const handleRerollGeneratedTrack = (mode, track, index) => {
    const generator = generators[mode]
    if (!generator || !generator.result) {
      return
    }
    const api = generatorApiMap[mode]
    if (!api) {
      return
    }
    const trackIds = generator.result.trackIds || []
    const resolvedIndex =
      typeof index === 'number'
        ? index
        : trackIds.findIndex((id) => id === (track?.id || track?.ID))
    if (resolvedIndex < 0 || resolvedIndex >= trackIds.length) {
      return
    }
    const trackId = trackIds[resolvedIndex] || track?.id || track?.ID
    if (!trackId) {
      return
    }
    const excludeSet = new Set(generator.exclude || [])
    trackIds.forEach((id) => {
      if (id) {
        excludeSet.add(id)
      }
    })
    excludeSet.add(trackId)
    setGenerators((prev) => ({
      ...prev,
      [mode]: {
        ...prev[mode],
        updatingTrackId: trackId,
      },
    }))
    const payload = {
      limit: 1,
      diversity: settings.baseDiversity,
      excludeTrackIds: Array.from(excludeSet),
    }
    if (mode === 'discovery') {
      payload.diversity = settings.discoveryExploration
    }
    api(payload)
      .then(({ data }) => {
        const newTrack = data?.tracks?.[0]
        const newTrackId = data?.trackIds?.[0]
        if (!newTrack || !newTrackId) {
          throw new Error(
            translate('pages.explore.rerollUnavailable', {
              _: 'No alternative songs available right now.',
            }),
          )
        }
        setGenerators((prev) => {
          const current = prev[mode]
          if (!current || !current.result) {
            return {
              ...prev,
              [mode]: {
                ...current,
                updatingTrackId: null,
              },
            }
          }
          const currentTrackIds = current.result.trackIds || []
          if (resolvedIndex < 0 || resolvedIndex >= currentTrackIds.length) {
            return {
              ...prev,
              [mode]: {
                ...current,
                updatingTrackId: null,
              },
            }
          }
          const nextTrackIds = [...currentTrackIds]
          const nextTracks = [...(current.result.tracks || [])]
          const previousId = currentTrackIds[resolvedIndex] || trackId
          nextTrackIds[resolvedIndex] = newTrackId
          nextTracks[resolvedIndex] = newTrack
          const nextExclude = new Set(current.exclude || [])
          if (previousId) {
            nextExclude.add(previousId)
          }
          return {
            ...prev,
            [mode]: {
              ...current,
              result: {
                ...current.result,
                trackIds: nextTrackIds,
                tracks: nextTracks,
              },
              exclude: Array.from(nextExclude),
              updatingTrackId: null,
            },
          }
        })
      })
      .catch((error) => {
        const message =
          error?.message ||
          translate('pages.explore.rerollFailed', {
            _: 'Unable to reroll this song. Please try again.',
          })
        notify(message, 'warning')
        setGenerators((prev) => ({
          ...prev,
          [mode]: {
            ...prev[mode],
            updatingTrackId: null,
          },
        }))
      })
  }

  const handleRemoveCustomTrack = (track, index) => {
    setCustomResult((prev) => {
      if (!prev) {
        return prev
      }
      const tracks = prev.tracks || []
      const trackIds = prev.trackIds || []
      const resolvedIndex =
        typeof index === 'number'
          ? index
          : trackIds.findIndex((id) => id === (track?.id || track?.ID))
      if (resolvedIndex < 0 || resolvedIndex >= trackIds.length) {
        return prev
      }
      const removedId = trackIds[resolvedIndex] || track?.id || track?.ID || ''
      const nextTracks = tracks.filter((_, idx) => idx !== resolvedIndex)
      const nextTrackIds = trackIds.filter((_, idx) => idx !== resolvedIndex)
      if (removedId) {
        setCustomExcludeIds((prevExclude) => {
          if (prevExclude.includes(removedId)) {
            return prevExclude
          }
          return [...prevExclude, removedId]
        })
      }
      return {
        ...prev,
        tracks: nextTracks,
        trackIds: nextTrackIds,
      }
    })
  }

  const handleRerollCustomTrack = (track, index) => {
    if (!customResult) {
      return
    }
    const trackIds = customResult.trackIds || []
    const resolvedIndex =
      typeof index === 'number'
        ? index
        : trackIds.findIndex((id) => id === (track?.id || track?.ID))
    if (resolvedIndex < 0 || resolvedIndex >= trackIds.length) {
      return
    }
    const trackId = trackIds[resolvedIndex] || track?.id || track?.ID
    if (!trackId) {
      return
    }
    const seeds = selectedSongs.map((song) => getSongId(song)).filter(Boolean)
    if (seeds.length === 0) {
      return
    }
    const excludeSet = new Set(customExcludeIds)
    trackIds.forEach((id) => {
      if (id) {
        excludeSet.add(id)
      }
    })
    excludeSet.add(trackId)
    setCustomUpdatingTrackId(trackId)
    dataProvider
      .getCustomRecommendations({
        songIds: seeds,
        limit: 1,
        diversity: settings.baseDiversity,
        excludeTrackIds: Array.from(excludeSet),
        excludePlaylistIds,
      })
      .then(({ data }) => {
        const newTrack = data?.tracks?.[0]
        const newTrackId = data?.trackIds?.[0]
        if (!newTrack || !newTrackId) {
          throw new Error(
            translate('pages.explore.rerollUnavailable', {
              _: 'No alternative songs available right now.',
            }),
          )
        }
        setCustomResult((prev) => {
          if (!prev) {
            return prev
          }
          const currentTrackIds = prev.trackIds || []
          if (resolvedIndex < 0 || resolvedIndex >= currentTrackIds.length) {
            return prev
          }
          const nextTrackIds = [...currentTrackIds]
          const nextTracks = [...(prev.tracks || [])]
          const previousId = currentTrackIds[resolvedIndex] || trackId
          nextTrackIds[resolvedIndex] = newTrackId
          nextTracks[resolvedIndex] = newTrack
          setCustomExcludeIds((prevExclude) => {
            const merged = new Set(prevExclude)
            if (previousId) {
              merged.add(previousId)
            }
            return Array.from(merged)
          })
          return {
            ...prev,
            trackIds: nextTrackIds,
            tracks: nextTracks,
          }
        })
      })
      .catch((error) => {
        const message =
          error?.message ||
          translate('pages.explore.rerollFailed', {
            _: 'Unable to reroll this song. Please try again.',
          })
        notify(message, 'warning')
      })
      .finally(() => {
        setCustomUpdatingTrackId(null)
      })
  }

  const handleGenerateCustom = () => {
    setCustomLoading(true)
    setCustomError(null)
    dataProvider
      .getCustomRecommendations({
        songIds: selectedSongs.map((song) => getSongId(song)).filter(Boolean),
        limit: settings.mixLength,
        diversity: settings.baseDiversity,
        excludePlaylistIds,
      })
      .then(({ data }) => {
        setCustomResult(data)
        setCustomName(
          data?.name ||
            translate('pages.explore.customDefaultName', { _: 'Custom Mix' }),
        )
        setCustomExcludeIds([])
        setCustomUpdatingTrackId(null)
      })
      .catch((error) => {
        const serverMessage =
          error?.body?.message ||
          error?.message ||
          translate('pages.explore.customNoSeeds', {
            _: 'Try selecting different songs and generate again.',
          })
        setCustomError(serverMessage)
        setCustomResult(null)
      })
      .finally(() => setCustomLoading(false))
  }

  const saveAsPlaylist = async (result, name) => {
    const ids = (result && result.trackIds) || []
    if (!result || ids.length === 0 || saving) {
      return
    }
    const playlistName = name && name.trim() ? name.trim() : result.name
    try {
      setSaving(true)
      const createResponse = await dataProvider.create('playlist', {
        data: { name: playlistName },
      })
      const playlistId = createResponse?.data?.id
      if (playlistId) {
        await dataProvider.create('playlistTrack', {
          data: { ids },
          filter: { playlist_id: playlistId },
        })
      }
      notify('pages.explore.playlistSaved', {
        type: 'info',
        messageArgs: { name: playlistName },
      })
      refresh()
    } catch (error) {
      notify(error?.message || 'ra.message.error', { type: 'warning' })
    } finally {
      setSaving(false)
    }
  }

  const pageTitle = translate('menu.explore.name', { _: 'Explore' })
  const playlistsLabel = translate('pages.explore.playlists', {
    _: 'Playlists for you',
  })
  const playlistCountLabel = translate('resources.playlist.fields.songCount', {
    _: 'Songs',
  })
  const updatedLabel = translate('resources.playlist.fields.updatedAt', {
    _: 'Updated at',
  })

  return (
    <Box className={classes.page}>
      <Title title={`${BRAND_NAME} - ${pageTitle}`} />
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        indicatorColor="primary"
        textColor="primary"
        className={classes.tabs}
      >
        <Tab
          label={translate('pages.explore.tabs.overview', { _: 'Overview' })}
        />
        <Tab
          label={translate('pages.explore.tabs.textGenerator', {
            _: 'Text Generator',
          })}
        />
        <Tab
          label={translate('pages.explore.tabs.settings', { _: 'Settings' })}
        />
      </Tabs>

      {activeTab === 0 && (
        <Box className={classes.tabPanel}>
          <Card className={classes.heroCard} variant="outlined">
            <ExploreIcon className={classes.heroIcon} />
            <Box>
              <Typography variant="h4">
                {translate('pages.explore.suggested', {
                  _: 'Explore Suggested',
                })}
              </Typography>
              <Typography variant="body1" className={classes.cardMeta}>
                {translate('pages.explore.suggestedSubtitle', {
                  _: 'Kick off your next listening session with a few hand-picked ideas.',
                })}
              </Typography>
            </Box>
          </Card>

          <Box className={classes.section}>
            {generatorConfigs.map((config) => renderGeneratorCard(config))}
          </Box>

          <Card className={classes.recommendationCard} variant="outlined">
            <Typography variant="h5">
              {translate('pages.explore.customTitle', {
                _: 'Create from selected songs',
              })}
            </Typography>
            <TextField
              variant="outlined"
              value={songQuery}
              onChange={(event) => setSongQuery(event.target.value)}
              placeholder={translate('pages.explore.searchPlaceholder', {
                _: 'Search for songs to add',
              })}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
                endAdornment: searchLoading ? (
                  <CircularProgress size={18} />
                ) : undefined,
              }}
            />
            {searchState.items.length > 0 && (
              <Paper className={classes.searchResults} variant="outlined">
                <List dense disablePadding>
                  {searchState.items.map((song) => {
                    const songId = song?.id || song?.ID
                    return (
                      <ListItem
                        key={songId || song?.title}
                        button
                        onClick={() => handleSelectSong(song)}
                      >
                        <ListItemText
                          primary={song.title}
                          secondary={`${song.artist || ''} • ${song.album || ''}`}
                        />
                      </ListItem>
                    )
                  })}
                </List>
                {searchState.hasMore && (
                  <Box className={classes.loadMoreContainer}>
                    <Button
                      variant="text"
                      onClick={handleLoadMoreSearch}
                      disabled={searchLoadingMore}
                    >
                      {searchLoadingMore ? (
                        <CircularProgress size={16} />
                      ) : (
                        translate('pages.explore.searchLoadMore', {
                          _: 'Load more results',
                        })
                      )}
                    </Button>
                  </Box>
                )}
              </Paper>
            )}
            <Box className={classes.selectedSongsContainer}>
              {selectedSongs.length === 0 && (
                <Typography variant="body2" className={classes.placeholder}>
                  {translate('pages.explore.selectedEmpty', {
                    _: 'No songs selected yet.',
                  })}
                </Typography>
              )}
              {selectedSongs.map((song, idx) => {
                const songId = getSongId(song)
                return (
                  <Chip
                    key={songId || idx}
                    label={`${song.title || ''} — ${song.artist || ''}`}
                    onDelete={() => handleRemoveSong(songId)}
                    color="primary"
                    variant="default"
                  />
                )
              })}
            </Box>
            <FormControl
              variant="outlined"
              className={classes.playlistExclusionControl}
              disabled={playlists.length === 0}
            >
              <InputLabel id="exclude-playlists-label">
                {translate('pages.explore.excludePlaylistsLabel', {
                  _: 'Exclude playlists',
                })}
              </InputLabel>
              <Select
                labelId="exclude-playlists-label"
                multiple
                value={excludePlaylistIds}
                onChange={(event) => {
                  const value = event.target.value
                  setExcludePlaylistIds(Array.isArray(value) ? value : [])
                }}
                label={translate('pages.explore.excludePlaylistsLabel', {
                  _: 'Exclude playlists',
                })}
                renderValue={(selected) => {
                  if (!selected || selected.length === 0) {
                    return translate(
                      'pages.explore.excludePlaylistsPlaceholder',
                      {
                        _: 'No playlists selected',
                      },
                    )
                  }
                  return (
                    <Box className={classes.selectChips}>
                      {selected.map((id) => {
                        const playlist = playlists.find((pls) => pls.id === id)
                        const name = playlist?.name || id
                        return (
                          <Chip
                            key={id}
                            label={name}
                            size="small"
                            className={classes.selectChip}
                          />
                        )
                      })}
                    </Box>
                  )
                }}
              >
                {playlists.map((playlist) => (
                  <MenuItem key={playlist.id} value={playlist.id}>
                    <Checkbox
                      checked={excludePlaylistIds.indexOf(playlist.id) > -1}
                    />
                    <ListItemText
                      primary={playlist.name}
                      secondary={playlist.comment || null}
                    />
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                {translate('pages.explore.excludePlaylistsHelper', {
                  _: 'Skip songs that appear in the selected playlists.',
                })}
              </FormHelperText>
            </FormControl>
            <Box className={classes.buttonRow}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleGenerateCustom}
                disabled={selectedSongs.length === 0 || customLoading}
              >
                {customLoading ? (
                  <CircularProgress size={18} color="inherit" />
                ) : (
                  translate('pages.explore.generateButton', {
                    _: 'Generate mix',
                  })
                )}
              </Button>
              {customError && (
                <Typography variant="body2" className={classes.error}>
                  {customError}
                </Typography>
              )}
            </Box>
            <RecommendationPreview
              result={customResult}
              playlistName={customName}
              onPlaylistNameChange={setCustomName}
              onSave={() => saveAsPlaylist(customResult, customName)}
              saving={saving}
              translate={translate}
              allowTrackActions
              onRemoveTrack={handleRemoveCustomTrack}
              onRerollTrack={handleRerollCustomTrack}
              pendingTrackId={customUpdatingTrackId}
            />
          </Card>

          <Box className={classes.section}>
            <Typography variant="h5">{playlistsLabel}</Typography>
            {playlistsLoading && <CircularProgress size={24} />}
            {playlistsError && !playlistsLoading && (
              <Typography variant="body2" className={classes.error}>
                {translate('pages.explore.playlistsError', {
                  _: 'We could not load playlists right now. Try again later.',
                })}
              </Typography>
            )}
            {playlistsLoaded && !playlistsError && playlists.length === 0 && (
              <Typography variant="body2" className={classes.placeholder}>
                {translate('pages.explore.playlistsEmpty', {
                  _: 'You do not have any playlists yet. Create one to see it here.',
                })}
              </Typography>
            )}
            {playlists.length > 0 && (
              <Box className={classes.grid}>
                {playlists.map((playlist) => (
                  <Card
                    key={playlist.id}
                    className={classes.card}
                    variant="outlined"
                  >
                    <CardActionArea
                      className={classes.cardAction}
                      component={Link}
                      to={`/playlist/${playlist.id}/show`}
                    >
                      <CardContent>
                        <Box className={classes.playlistHeader}>
                          <QueueMusicIcon color="primary" />
                          <Typography variant="h6">{playlist.name}</Typography>
                        </Box>
                        {playlist.comment && (
                          <Typography
                            variant="body2"
                            className={classes.cardMeta}
                          >
                            {playlist.comment}
                          </Typography>
                        )}
                        <Typography
                          variant="body2"
                          className={classes.cardMeta}
                        >
                          {`${playlistCountLabel}: ${playlist.songCount ?? 0}`}
                        </Typography>
                        {playlist.updatedAt && (
                          <Typography
                            variant="caption"
                            className={classes.cardMeta}
                          >
                            {`${updatedLabel}: ${new Date(
                              playlist.updatedAt,
                            ).toLocaleString()}`}
                          </Typography>
                        )}
                      </CardContent>
                    </CardActionArea>
                  </Card>
                ))}
              </Box>
            )}
          </Box>
        </Box>
      )}

      {activeTab === 1 && (
        <Box className={classes.tabPanel}>
          <TextPlaylistGenerator />
        </Box>
      )}

      {activeTab === 2 && (
        <Box className={classes.tabPanel}>
          <ExploreSettingsPanel
            translate={translate}
            draft={settingsDraft}
            errors={settingsErrors}
            onFieldChange={handleSettingsFieldChange}
            onReset={handleSettingsReset}
            onSave={handleSettingsSave}
            saving={settingsSaving}
            dirty={settingsDirty}
            loading={settingsLoading}
            serverError={settingsMessage}
          />
        </Box>
      )}
    </Box>
  )
}

export default ExploreSuggestions
