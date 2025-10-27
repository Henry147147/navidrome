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

  const [recentResult, setRecentResult] = useState(null)
  const [recentName, setRecentName] = useState('')
  const [recentLoading, setRecentLoading] = useState(false)
  const [recentError, setRecentError] = useState(null)

  const [customResult, setCustomResult] = useState(null)
  const [customName, setCustomName] = useState('')
  const [customLoading, setCustomLoading] = useState(false)
  const [customError, setCustomError] = useState(null)

  const [saving, setSaving] = useState(false)

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

  const handleGenerateRecent = () => {
    setRecentLoading(true)
    setRecentError(null)
    dataProvider
      .getRecentRecommendations({ limit: DEFAULT_RECOMMENDATION_LIMIT })
      .then(({ data }) => {
        setRecentResult(data)
        setRecentName(data?.name || translate('pages.explore.recentDefaultName', { _: 'Recent Mix' }))
      })
      .catch((error) => {
        const serverMessage =
          error?.body?.message ||
          error?.message ||
          translate('pages.explore.noRecentSeeds', {
            _: 'Play a few songs and try again.',
          })
        setRecentError(serverMessage)
        setRecentResult(null)
      })
      .finally(() => setRecentLoading(false))
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

      <Card className={classes.recommendationCard} variant="outlined">
        <Typography variant="h5">
          {translate('pages.explore.recentTitle', { _: 'Generate from recent listens' })}
        </Typography>
        <Typography variant="body2" className={classes.placeholder}>
          {translate('pages.explore.recentDescription', {
            _: 'Build a personalised mix using the tracks you have been enjoying lately.',
          })}
        </Typography>
        <Box className={classes.buttonRow}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleGenerateRecent}
            disabled={recentLoading}
          >
            {recentLoading ? (
              <CircularProgress size={18} color="inherit" />
            ) : (
              translate('pages.explore.generateButton', { _: 'Generate mix' })
            )}
          </Button>
          {recentError && (
            <Typography variant="body2" className={classes.error}>
              {recentError}
            </Typography>
          )}
        </Box>
        <RecommendationPreview
          result={recentResult}
          playlistName={recentName}
          onPlaylistNameChange={setRecentName}
          onSave={() => saveAsPlaylist(recentResult, recentName)}
          saving={saving}
          translate={translate}
        />
      </Card>

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
