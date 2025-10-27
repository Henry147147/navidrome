import React, { useEffect, useMemo, useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardActionArea,
  CardContent,
  CircularProgress,
  Divider,
  InputAdornment,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Paper,
  TextField,
  Typography,
  Chip,
  Slider,
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
import config from '../config'
import { BRAND_NAME } from '../consts'
import { formatDuration } from '../utils'

const useStyles = makeStyles((theme) => ({
  page: {
    marginTop: theme.spacing(2),
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
  sliderRow: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(2),
    flexWrap: 'wrap',
  },
  searchResults: {
    marginTop: theme.spacing(1),
    maxHeight: 260,
    overflowY: 'auto',
  },
  selectedSongsContainer: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(1),
  },
  listItemPrimary: {
    fontWeight: theme.typography.fontWeightMedium,
  },
}))

const PLAYLIST_LIMIT = Math.min(6, config.maxSidebarPlaylists || 6)
const DEFAULT_RECOMMENDATION_LIMIT = 25

const RecommendationPreview = ({
  result,
  playlistName,
  onPlaylistNameChange,
  onSave,
  saving,
  translate,
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
          {tracks.map((track, index) => (
            <React.Fragment key={track.id || track.ID || index}>
              <ListItem>
                <ListItemText
                  primary={
                    <span className={classes.listItemPrimary}>
                      {track.title}
                    </span>
                  }
                  secondary={`${track.artist || ''} • ${track.album || ''}`}
                />
                {track.duration && (
                  <Typography variant="caption">
                    {formatDuration(track.duration)}
                  </Typography>
                )}
              </ListItem>
              {index < tracks.length - 1 && <Divider component="li" />}
            </React.Fragment>
          ))}
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

	const [generators, setGenerators] = useState(() => ({
		recent: { result: null, name: '', loading: false, error: null },
		favorites: { result: null, name: '', loading: false, error: null },
		all: { result: null, name: '', loading: false, error: null },
		discovery: { result: null, name: '', loading: false, error: null },
	}))
	const [discoveryDiversity, setDiscoveryDiversity] = useState(0.6)
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
	const updateGeneratorName = (mode, value) => {
		setGenerators((prev) => ({
			...prev,
			[mode]: { ...prev[mode], name: value },
		}))
	}
	const runGenerator = (mode, requestOptions = {}, defaultNameKey, errorKey) => {
		const api = generatorApiMap[mode]
		if (!api) {
			return
		}
		setGenerators((prev) => ({
			...prev,
			[mode]: { ...prev[mode], loading: true, error: null },
		}))
		api({ limit: DEFAULT_RECOMMENDATION_LIMIT, ...requestOptions })
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
				_: 'Surface songs that are farther from your usual listening. Adjust the exploration slider.',
			}),
			defaultNameKey: 'pages.explore.discoveryDefaultName',
			errorKey: 'pages.explore.noDiscoverySeeds',
			slider: true,
			onGenerate: () =>
				runGenerator(
					'discovery',
					{ diversity: discoveryDiversity },
					'pages.explore.discoveryDefaultName',
					'pages.explore.noDiscoverySeeds',
				),
		},
	]
	const renderGeneratorCard = (config) => {
		const state =
			generators[config.key] || {
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
				{config.slider && (
					<Box className={classes.sliderRow}>
						<Typography variant="body2">
							{translate('pages.explore.discoverySliderLabel', {
								_: 'Exploration',
							})}
						</Typography>
						<Slider
							value={discoveryDiversity}
							onChange={(_, value) => {
								const nextValue = Array.isArray(value) ? value[0] : value
								setDiscoveryDiversity(nextValue)
							}}
							min={0}
							max={1}
							step={0.05}
							valueLabelDisplay="auto"
							valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
						/>
						<Typography variant="body2" className={classes.placeholder}>
							{translate('pages.explore.discoverySliderValue', {
								_: '%{value}% exploratory',
								value: Math.round(discoveryDiversity * 100),
							})}
						</Typography>
						<Typography variant="caption" className={classes.placeholder}>
							{translate('pages.explore.discoverySliderHelper', {
								_: 'Higher values lean toward more adventurous picks.',
							})}
						</Typography>
					</Box>
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
					onPlaylistNameChange={(value) => updateGeneratorName(config.key, value)}
					onSave={() => saveAsPlaylist(state.result, state.name)}
					saving={saving}
					translate={translate}
				/>
			</Card>
		)
	}

	const [customResult, setCustomResult] = useState(null)
	const [customName, setCustomName] = useState('')
	const [customLoading, setCustomLoading] = useState(false)
	const [customError, setCustomError] = useState(null)

  const [songQuery, setSongQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [selectedSongs, setSelectedSongs] = useState([])

  useEffect(() => {
    if (!songQuery.trim()) {
      setSearchResults([])
      return
    }

    const trimmed = songQuery.trim()
    setSearchLoading(true)
    const timer = setTimeout(() => {
      dataProvider
        .getList('song', {
          pagination: { page: 1, perPage: 10 },
          sort: { field: 'title', order: 'ASC' },
          filter: { q: trimmed },
        })
        .then((response) => {
          setSearchResults(response.data || [])
        })
        .catch(() => setSearchResults([]))
        .finally(() => setSearchLoading(false))
    }, 250)

    return () => clearTimeout(timer)
  }, [songQuery, dataProvider])

  const handleSelectSong = (song) => {
    if (!song || !song.id) {
      return
    }
    setSelectedSongs((prev) => {
      if (prev.some((item) => item.id === song.id)) {
        return prev
      }
      return [...prev, song]
    })
    setSongQuery('')
    setSearchResults([])
  }

  const handleRemoveSong = (id) => {
    setSelectedSongs((prev) => prev.filter((song) => song.id !== id))
  }

	const handleGenerateCustom = () => {
		setCustomLoading(true)
		setCustomError(null)
		dataProvider
			.getCustomRecommendations({
        songIds: selectedSongs.map((song) => song.id),
        limit: DEFAULT_RECOMMENDATION_LIMIT,
      })
      .then(({ data }) => {
        setCustomResult(data)
        setCustomName(
          data?.name || translate('pages.explore.customDefaultName', { _: 'Custom Mix' }),
        )
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
        {searchResults.length > 0 && (
          <Paper className={classes.searchResults} variant="outlined">
            {searchResults.map((song) => (
              <MenuItem key={song.id} onClick={() => handleSelectSong(song)}>
                <ListItemText
                  primary={song.title}
                  secondary={`${song.artist || ''} • ${song.album || ''}`}
                />
              </MenuItem>
            ))}
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
          {selectedSongs.map((song) => (
            <Chip
              key={song.id}
              label={`${song.title || ''} — ${song.artist || ''}`}
              onDelete={() => handleRemoveSong(song.id)}
              color="primary"
              variant="default"
            />
          ))}
        </Box>
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
              translate('pages.explore.generateButton', { _: 'Generate mix' })
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
              <Card key={playlist.id} className={classes.card} variant="outlined">
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
                      <Typography variant="body2" className={classes.cardMeta}>
                        {playlist.comment}
                      </Typography>
                    )}
                    <Typography variant="body2" className={classes.cardMeta}>
                      {`${playlistCountLabel}: ${playlist.songCount ?? 0}`}
                    </Typography>
                    {playlist.updatedAt && (
                      <Typography variant="caption" className={classes.cardMeta}>
                        {`${updatedLabel}: ${new Date(playlist.updatedAt).toLocaleString()}`}
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
  )
}

export default ExploreSuggestions
