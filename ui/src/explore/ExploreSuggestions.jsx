import React, { useMemo } from 'react'
import {
  Box,
  Card,
  CardActionArea,
  CardContent,
  CircularProgress,
  Typography,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import { Title, useQueryWithStore, useTranslate } from 'react-admin'
import { Link } from 'react-router-dom'
import ExploreIcon from '@material-ui/icons/Explore'
import QueueMusicIcon from '@material-ui/icons/QueueMusic'
import config from '../config'
import { BRAND_NAME } from '../consts'

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
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
    gap: theme.spacing(2),
  },
  card: {
    height: '100%',
  },
  cardMeta: {
    color: theme.palette.text.secondary,
    marginTop: theme.spacing(1),
  },
  cardAction: {
    height: '100%',
    alignItems: 'stretch',
  },
  playlistHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
  },
  placeholder: {
    color: theme.palette.text.secondary,
  },
  error: {
    color: theme.palette.error.main,
  },
}))

const SUGGESTED_COLLECTIONS = [
  {
    id: 'daily_drive',
    title: 'Daily Drive',
    description: 'Fresh tracks blended with timeless favourites for your commute.',
  },
  {
    id: 'focus_flow',
    title: 'Focus Flow',
    description: 'Instrumental downtempo and ambient pulses to keep you in the zone.',
  },
  {
    id: 'weekend_warmup',
    title: 'Weekend Warm-up',
    description: 'Feel-good pop and disco cuts to kick off the evening.',
  },
  {
    id: 'deep_crates',
    title: 'Deep Crates',
    description: 'Underground discoveries spanning jazz, soul, and sample-ready grooves.',
  },
]

const HISTORY_RECOMMENDATIONS = [
  {
    id: 'everlong',
    song: 'Everlong',
    items: [
      {
        id: 'post_grunge',
        title: 'Post-Grunge Staples',
        description: 'An electric mix of late-90s anthems and roaring guitars.',
      },
      {
        id: 'alt_rock_club',
        title: 'Alt Rock Club',
        description: 'Crank up the distortion with modern riff-heavy favourites.',
      },
      {
        id: 'unplugged_evenings',
        title: 'Unplugged Evenings',
        description: 'Laid-back acoustic takes from your favourite rock outfits.',
      },
    ],
  },
  {
    id: 'bad_guy',
    song: 'bad guy',
    items: [
      {
        id: 'dark_pop',
        title: 'Dark Pop Currents',
        description: 'Shadowy electro-pop driven by heavy bass and whisper vocals.',
      },
      {
        id: 'hyper_modern',
        title: 'Hyper Modern',
        description: 'Glitchy beats and experimental textures on the pop frontier.',
      },
      {
        id: 'late_night',
        title: 'Late Night Loops',
        description: 'Minimal electronic tracks for after-hours listening.',
      },
    ],
  },
  {
    id: 'harvest_moon',
    song: 'Harvest Moon',
    items: [
      {
        id: 'campfire',
        title: 'Campfire Classics',
        description: 'Americana favourites and gentle sing-alongs.',
      },
      {
        id: 'coffeehouse',
        title: 'Coffeehouse Morning',
        description: 'Soft folk and indie ballads for slow starts.',
      },
      {
        id: 'roots_revival',
        title: 'Roots Revival',
        description: 'Modern storytellers keeping folk traditions alive.',
      },
    ],
  },
]

const PLAYLIST_LIMIT = Math.min(6, config.maxSidebarPlaylists || 6)

const ExploreSuggestions = () => {
  const classes = useStyles()
  const translate = useTranslate()
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

      <Box className={classes.section}>
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
        <Box className={classes.grid}>
          {SUGGESTED_COLLECTIONS.map((collection) => (
            <Card key={collection.id} className={classes.card} variant="outlined">
              <CardContent>
                <Typography variant="h6">{collection.title}</Typography>
                <Typography variant="body2" className={classes.cardMeta}>
                  {collection.description}
                </Typography>
              </CardContent>
            </Card>
          ))}
        </Box>
      </Box>

      {HISTORY_RECOMMENDATIONS.map((group) => (
        <Box key={group.id} className={classes.section}>
          <Typography variant="h5">
            {translate('pages.explore.becauseListened', {
              song: group.song,
              _: `Because you listened to "${group.song}" recently`,
            })}
          </Typography>
          <Box className={classes.grid}>
            {group.items.map((item) => (
              <Card key={item.id} className={classes.card} variant="outlined">
                <CardContent>
                  <Typography variant="h6">{item.title}</Typography>
                  <Typography variant="body2" className={classes.cardMeta}>
                    {item.description}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Box>
        </Box>
      ))}

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
