import React, { useCallback, useMemo, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import {
  useDataProvider,
  useGetList,
  useGetOne,
  useNotify,
  useTranslate,
} from 'react-admin'
import { GlobalHotKeys } from 'react-hotkeys'
import IconButton from '@material-ui/core/IconButton'
import {
  FormControl,
  FormControlLabel,
  InputLabel,
  Menu,
  Select,
  Switch,
  Typography,
  useMediaQuery,
} from '@material-ui/core'
import { RiSaveLine } from 'react-icons/ri'
import AutorenewIcon from '@material-ui/icons/Autorenew'
import TuneIcon from '@material-ui/icons/Tune'
import { LoveButton, useToggleLove } from '../common'
import {
  openSaveQueueDialog,
  setAutoPlayEnabled,
  setStreamingOverride,
  syncAutoPlaySettings,
} from '../actions'
import { keyMap } from '../hotkeys'
import { makeStyles } from '@material-ui/core/styles'
import { BITRATE_CHOICES, DEFAULT_SHARE_BITRATE } from '../consts'
import { DEFAULT_STREAMING_OVERRIDE } from './streamingOverrideUtils'
import { normalizeAutoPlaySettings } from '../autoplay/runtime'

const STREAM_PROFILE_DEFAULT = '__default__'

const useStyles = makeStyles((theme) => ({
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    flexGrow: 1,
    justifyContent: 'flex-end',
    gap: '0.5rem',
    listStyle: 'none',
    padding: 0,
    margin: 0,
  },
  mobileListItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    listStyle: 'none',
    padding: theme.spacing(0.5),
    margin: 0,
    height: 24,
  },
  button: {
    width: '2.5rem',
    height: '2.5rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
  },
  mobileButton: {
    width: 24,
    height: 24,
    padding: 0,
    margin: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '18px',
  },
  mobileIcon: {
    fontSize: '18px',
    display: 'flex',
    alignItems: 'center',
  },
  streamMenuContent: {
    padding: theme.spacing(2),
    minWidth: 260,
    display: 'grid',
    gap: theme.spacing(2),
  },
  streamMenuHint: {
    color: theme.palette.text.secondary,
    display: 'block',
  },
}))

const PlayerToolbar = ({ id, isRadio }) => {
  const dispatch = useDispatch()
  const translate = useTranslate()
  const notify = useNotify()
  const dataProvider = useDataProvider()
  const streamingOverride = useSelector(
    (state) => state.settings?.streamingOverride || DEFAULT_STREAMING_OVERRIDE,
  )
  const autoplay = useSelector((state) => state.autoplay || { enabled: false })
  const { data, loading } = useGetOne('song', id, { enabled: !!id && !isRadio })
  const { data: transcodingData = {}, loading: loadingTranscodings } =
    useGetList(
      'transcoding',
      {
        page: 1,
        perPage: 1000,
      },
      { field: 'name', order: 'ASC' },
    )
  const [toggleLove, toggling] = useToggleLove('song', data)
  const [streamMenuAnchor, setStreamMenuAnchor] = useState(null)
  const isDesktop = useMediaQuery('(min-width:810px)')
  const classes = useStyles()

  const transcodings = useMemo(
    () => Object.values(transcodingData),
    [transcodingData],
  )

  const transcodingById = useMemo(() => {
    return transcodings.reduce((acc, profile) => {
      if (!profile?.id) {
        return acc
      }
      acc[profile.id] = profile
      return acc
    }, {})
  }, [transcodings])

  const currentProfileId =
    streamingOverride.mode === 'override' && streamingOverride.profileId
      ? streamingOverride.profileId
      : STREAM_PROFILE_DEFAULT
  const effectiveProfileId =
    currentProfileId !== STREAM_PROFILE_DEFAULT &&
    transcodingById[currentProfileId]
      ? currentProfileId
      : STREAM_PROFILE_DEFAULT
  const currentBitrate =
    streamingOverride.mode === 'override' && streamingOverride.maxBitRate
      ? Number(streamingOverride.maxBitRate)
      : DEFAULT_SHARE_BITRATE
  const forceTranscoding =
    effectiveProfileId !== STREAM_PROFILE_DEFAULT &&
    streamingOverride.mode === 'override' &&
    Boolean(streamingOverride.forceTranscoding)

  const handlers = {
    TOGGLE_LOVE: useCallback(() => toggleLove(), [toggleLove]),
  }

  const handleSaveQueue = useCallback(
    (e) => {
      dispatch(openSaveQueueDialog())
      e.stopPropagation()
    },
    [dispatch],
  )

  const handleAutoPlayToggle = useCallback(async () => {
    const previousSettings = normalizeAutoPlaySettings({
      enabled: Boolean(autoplay.enabled),
      mode: autoplay.mode,
      textPrompt: autoplay.textPrompt,
      excludePlaylistIds: autoplay.excludePlaylistIds,
      batchSize: autoplay.batchSize,
      diversityOverride: autoplay.diversityOverride,
    })
    const nextEnabled = !previousSettings.enabled
    dispatch(setAutoPlayEnabled(nextEnabled))

    try {
      const { data } = await dataProvider.updateAutoPlaySettings({
        ...previousSettings,
        enabled: nextEnabled,
      })
      dispatch(syncAutoPlaySettings(data))
      notify('pages.autoplay.settings.saved', { type: 'info' })
    } catch (e) {
      dispatch(syncAutoPlaySettings(previousSettings))
      notify('pages.autoplay.settings.serverError', { type: 'warning' })
    }
  }, [autoplay, dataProvider, dispatch, notify])

  const openStreamSettings = useCallback((e) => {
    e.stopPropagation()
    setStreamMenuAnchor(e.currentTarget)
  }, [])

  const closeStreamSettings = useCallback((e) => {
    e?.stopPropagation?.()
    setStreamMenuAnchor(null)
  }, [])

  const handleProfileChange = useCallback(
    (e) => {
      const profileId = e.target.value
      if (profileId === STREAM_PROFILE_DEFAULT) {
        dispatch(setStreamingOverride(DEFAULT_STREAMING_OVERRIDE))
        return
      }

      const profile = transcodingById[profileId]
      if (!profile) {
        dispatch(setStreamingOverride(DEFAULT_STREAMING_OVERRIDE))
        return
      }

      const selectedBitrate =
        Number(profile.defaultBitRate) || DEFAULT_SHARE_BITRATE
      dispatch(
        setStreamingOverride({
          mode: 'override',
          forceTranscoding,
          profileId,
          format: profile.targetFormat,
          maxBitRate: selectedBitrate,
        }),
      )
    },
    [dispatch, forceTranscoding, transcodingById],
  )

  const handleBitrateChange = useCallback(
    (e) => {
      if (effectiveProfileId === STREAM_PROFILE_DEFAULT) {
        return
      }
      const profile = transcodingById[effectiveProfileId]
      if (!profile) {
        dispatch(setStreamingOverride(DEFAULT_STREAMING_OVERRIDE))
        return
      }
      dispatch(
        setStreamingOverride({
          mode: 'override',
          forceTranscoding,
          profileId: effectiveProfileId,
          format: profile.targetFormat,
          maxBitRate: Number(e.target.value),
        }),
      )
    },
    [effectiveProfileId, dispatch, forceTranscoding, transcodingById],
  )

  const handleForceToggle = useCallback(
    (e) => {
      const checked = Boolean(e.target.checked)
      if (effectiveProfileId === STREAM_PROFILE_DEFAULT) {
        dispatch(setStreamingOverride(DEFAULT_STREAMING_OVERRIDE))
        return
      }
      const profile = transcodingById[effectiveProfileId]
      if (!profile) {
        dispatch(setStreamingOverride(DEFAULT_STREAMING_OVERRIDE))
        return
      }
      dispatch(
        setStreamingOverride({
          mode: 'override',
          forceTranscoding: checked,
          profileId: effectiveProfileId,
          format: profile.targetFormat,
          maxBitRate: currentBitrate,
        }),
      )
    },
    [currentBitrate, dispatch, effectiveProfileId, transcodingById],
  )

  const buttonClass = isDesktop ? classes.button : classes.mobileButton
  const listItemClass = isDesktop ? classes.toolbar : classes.mobileListItem

  const saveQueueButton = (
    <IconButton
      size={isDesktop ? 'small' : undefined}
      onClick={handleSaveQueue}
      disabled={isRadio}
      data-testid="save-queue-button"
      className={buttonClass}
    >
      <RiSaveLine className={!isDesktop ? classes.mobileIcon : undefined} />
    </IconButton>
  )

  const autoplayButton = (
    <IconButton
      size={isDesktop ? 'small' : undefined}
      onClick={handleAutoPlayToggle}
      data-testid="autoplay-toggle-button"
      className={buttonClass}
      color={autoplay.enabled ? 'primary' : 'default'}
      title={translate('player.autoplayToggleText')}
      aria-label={translate(
        autoplay.enabled
          ? 'player.autoplayEnabledText'
          : 'player.autoplayDisabledText',
      )}
    >
      <AutorenewIcon className={!isDesktop ? classes.mobileIcon : undefined} />
    </IconButton>
  )

  const streamSettingsButton = (
    <>
      <IconButton
        size={isDesktop ? 'small' : undefined}
        onClick={openStreamSettings}
        disabled={isRadio || !id}
        data-testid="stream-settings-button"
        className={buttonClass}
        title={translate('player.streamSettingsText')}
      >
        <TuneIcon className={!isDesktop ? classes.mobileIcon : undefined} />
      </IconButton>
      <Menu
        anchorEl={streamMenuAnchor}
        keepMounted
        open={Boolean(streamMenuAnchor)}
        onClose={closeStreamSettings}
      >
        <div
          className={classes.streamMenuContent}
          onClick={(e) => e.stopPropagation()}
        >
          <Typography variant="caption" className={classes.streamMenuHint}>
            {translate('player.streamDefaultBehaviorText')}
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={forceTranscoding}
                onChange={handleForceToggle}
                color="primary"
                disabled={effectiveProfileId === STREAM_PROFILE_DEFAULT}
                inputProps={{ 'data-testid': 'stream-force-toggle' }}
              />
            }
            label={translate('player.streamForceTranscodingText')}
          />
          <FormControl variant="outlined" size="small" fullWidth>
            <InputLabel id="stream-profile-select-label">
              {translate('player.streamProfileText')}
            </InputLabel>
            <Select
              native
              labelId="stream-profile-select-label"
              value={effectiveProfileId}
              onChange={handleProfileChange}
              label={translate('player.streamProfileText')}
              inputProps={{ 'data-testid': 'stream-profile-select' }}
              disabled={loadingTranscodings}
            >
              <option value={STREAM_PROFILE_DEFAULT}>
                {translate('player.streamDefaultBehaviorText')}
              </option>
              {transcodings.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </Select>
          </FormControl>
          <FormControl variant="outlined" size="small" fullWidth>
            <InputLabel id="stream-bitrate-select-label">
              {translate('player.streamBitrateText')}
            </InputLabel>
            <Select
              native
              labelId="stream-bitrate-select-label"
              value={currentBitrate}
              onChange={handleBitrateChange}
              label={translate('player.streamBitrateText')}
              inputProps={{ 'data-testid': 'stream-bitrate-select' }}
              disabled={effectiveProfileId === STREAM_PROFILE_DEFAULT}
            >
              {BITRATE_CHOICES.map((choice) => (
                <option key={choice.id} value={choice.id}>
                  {choice.name}
                </option>
              ))}
            </Select>
          </FormControl>
        </div>
      </Menu>
    </>
  )

  const loveButton = (
    <LoveButton
      record={data}
      resource={'song'}
      size={isDesktop ? undefined : 'inherit'}
      disabled={loading || toggling || !id || isRadio}
      className={buttonClass}
    />
  )

  return (
    <>
      <GlobalHotKeys keyMap={keyMap} handlers={handlers} allowChanges />
      {isDesktop ? (
        <li className={`${listItemClass} item`}>
          {saveQueueButton}
          {autoplayButton}
          {streamSettingsButton}
          {loveButton}
        </li>
      ) : (
        <>
          <li className={`${listItemClass} item`}>{saveQueueButton}</li>
          <li className={`${listItemClass} item`}>{autoplayButton}</li>
          <li className={`${listItemClass} item`}>{streamSettingsButton}</li>
          <li className={`${listItemClass} item`}>{loveButton}</li>
        </>
      )}
    </>
  )
}

export default PlayerToolbar
