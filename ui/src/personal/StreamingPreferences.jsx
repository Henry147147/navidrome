import { useMemo } from 'react'
import { useGetList, useTranslate } from 'react-admin'
import { useDispatch, useSelector } from 'react-redux'
import {
  FormControl,
  FormControlLabel,
  FormHelperText,
  InputLabel,
  Select,
  Switch,
} from '@material-ui/core'
import { setStreamingOverride } from '../actions'
import { BITRATE_CHOICES, DEFAULT_SHARE_BITRATE } from '../consts'
import { DEFAULT_STREAMING_OVERRIDE } from '../audioplayer/streamingOverrideUtils'

const STREAM_PROFILE_DEFAULT = '__default__'

export const StreamingPreferences = () => {
  const translate = useTranslate()
  const dispatch = useDispatch()
  const streamingOverride = useSelector(
    (state) => state.settings?.streamingOverride || DEFAULT_STREAMING_OVERRIDE,
  )
  const { data: transcodingData = {}, loading: loadingTranscodings } =
    useGetList(
      'transcoding',
      {
        page: 1,
        perPage: 1000,
      },
      { field: 'name', order: 'ASC' },
    )

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

  const onProfileChange = (event) => {
    const profileId = event.target.value
    if (profileId === STREAM_PROFILE_DEFAULT) {
      dispatch(setStreamingOverride(DEFAULT_STREAMING_OVERRIDE))
      return
    }

    const profile = transcodingById[profileId]
    if (!profile) {
      dispatch(setStreamingOverride(DEFAULT_STREAMING_OVERRIDE))
      return
    }

    dispatch(
      setStreamingOverride({
        mode: 'override',
        forceTranscoding,
        profileId,
        format: profile.targetFormat,
        maxBitRate:
          Number(streamingOverride.maxBitRate) ||
          Number(profile.defaultBitRate) ||
          DEFAULT_SHARE_BITRATE,
      }),
    )
  }

  const onBitrateChange = (event) => {
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
        maxBitRate: Number(event.target.value),
      }),
    )
  }

  const onForceToggle = (event) => {
    const checked = Boolean(event.target.checked)
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
  }

  return (
    <>
      <FormControl variant="outlined" margin="dense">
        <InputLabel htmlFor="personal-stream-profile-select">
          {translate('menu.personal.options.streamingProfile')}
        </InputLabel>
        <Select
          native
          id="personal-stream-profile-select"
          value={effectiveProfileId}
          onChange={onProfileChange}
          label={translate('menu.personal.options.streamingProfile')}
          disabled={loadingTranscodings}
          inputProps={{ 'data-testid': 'personal-stream-profile-select' }}
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

      <FormControl variant="outlined" margin="dense">
        <InputLabel htmlFor="personal-stream-bitrate-select">
          {translate('menu.personal.options.streamingBitrate')}
        </InputLabel>
        <Select
          native
          id="personal-stream-bitrate-select"
          value={currentBitrate}
          onChange={onBitrateChange}
          label={translate('menu.personal.options.streamingBitrate')}
          disabled={effectiveProfileId === STREAM_PROFILE_DEFAULT}
          inputProps={{ 'data-testid': 'personal-stream-bitrate-select' }}
        >
          {BITRATE_CHOICES.map((choice) => (
            <option key={choice.id} value={choice.id}>
              {choice.name}
            </option>
          ))}
        </Select>
      </FormControl>

      <FormControl>
        <FormControlLabel
          control={
            <Switch
              id="personal-force-transcoding"
              color="primary"
              checked={forceTranscoding}
              disabled={effectiveProfileId === STREAM_PROFILE_DEFAULT}
              onChange={onForceToggle}
              inputProps={{ 'data-testid': 'personal-stream-force-toggle' }}
            />
          }
          label={<span>{translate('menu.personal.options.forceTranscoding')}</span>}
        />
        <FormHelperText>
          {translate('menu.personal.options.streamingHelpText')}
        </FormHelperText>
      </FormControl>
    </>
  )
}
