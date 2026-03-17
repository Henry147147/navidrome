import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { useMediaQuery } from '@material-ui/core'
import { ThemeProvider } from '@material-ui/core/styles'
import {
  createMuiTheme,
  useAuthState,
  useDataProvider,
  useTranslate,
} from 'react-admin'
import ReactGA from 'react-ga'
import { GlobalHotKeys } from 'react-hotkeys'
import ReactJkMusicPlayer from 'navidrome-music-player'
import 'navidrome-music-player/assets/index.css'
import useCurrentTheme from '../themes/useCurrentTheme'
import config from '../config'
import useStyle from './styles'
import AudioTitle from './AudioTitle'
import {
  clearQueue,
  currentPlaying,
  markAutoPlayTrackPlayed,
  refreshQueue,
  setPlayMode,
  setTranscodingProfile,
  syncAutoPlaySettings,
  setVolume,
  syncQueue,
} from '../actions'
import PlayerToolbar from './PlayerToolbar'
import { sendNotification } from '../utils'
import subsonic from '../subsonic'
import locale from './locale'
import { keyMap } from '../hotkeys'
import keyHandlers from './keyHandlers'
import { calculateGain } from '../utils/calculateReplayGain'
import { BRAND_NAME } from '../consts'
import {
  buildOverrideKey,
  decorateQueueWithOverride,
  DEFAULT_STREAMING_OVERRIDE,
  toStreamQuery,
} from './streamingOverrideUtils'
import { detectBrowserProfile, decisionService } from '../transcode'
import {
  getQueueItemTrackId,
  getRemainingQueue,
  refillAutoPlayQueue,
} from '../autoplay/runtime'

const Player = () => {
  const theme = useCurrentTheme()
  const translate = useTranslate()
  const playerTheme = theme.player?.theme || 'dark'
  const dataProvider = useDataProvider()
  const playerState = useSelector((state) => state.player)
  const autoplayState = useSelector((state) => state.autoplay)
  const dispatch = useDispatch()
  const [startTime, setStartTime] = useState(null)
  const [scrobbled, setScrobbled] = useState(false)
  const [preloaded, setPreload] = useState(false)
  const [audioInstance, setAudioInstance] = useState(null)
  const isDesktop = useMediaQuery('(min-width:810px)')
  const isMobilePlayer =
    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent,
    )

  const { authenticated } = useAuthState()

  // Keep a ref to playerState so the mount effect can read the latest value
  // without re-triggering on every queue/position change
  const playerStateRef = useRef(playerState)
  playerStateRef.current = playerState

  // Detect browser codec profile and eagerly resolve transcode URLs for the
  // persisted queue once on mount (e.g. after a browser refresh)
  useEffect(() => {
    const profile = detectBrowserProfile()
    decisionService.setProfile(profile)
    dispatch(setTranscodingProfile(profile))

    const state = playerStateRef.current
    const currentIdx = state.savedPlayIndex || 0
    const trackIds = state.queue
      .slice(currentIdx, currentIdx + 4)
      .filter((item) => !item.isRadio && item.trackId)
      .map((item) => item.trackId)

    if (trackIds.length === 0) {
      dispatch(refreshQueue())
      return
    }

    Promise.allSettled(
      trackIds.map((id) =>
        decisionService.resolveStreamUrl(id).then((url) => [id, url]),
      ),
    ).then((results) => {
      const resolvedUrls = {}
      results.forEach((r) => {
        if (r.status === 'fulfilled') {
          resolvedUrls[r.value[0]] = r.value[1]
        }
      })
      dispatch(refreshQueue(resolvedUrls))
    })
  }, [dispatch])

  useEffect(() => {
    let active = true
    if (typeof dataProvider?.getAutoPlaySettings !== 'function') {
      return () => {
        active = false
      }
    }

    dataProvider
      .getAutoPlaySettings()
      .then(({ data }) => {
        if (active) {
          dispatch(syncAutoPlaySettings(data))
        }
      })
      .catch(() => {})

    return () => {
      active = false
    }
  }, [dataProvider, dispatch])

  // Pre-fetch transcode decisions for next 2-3 songs when queue or position changes
  useEffect(() => {
    if (!playerState.queue.length) return

    const currentIdx = playerState.savedPlayIndex || 0
    const nextSongIds = playerState.queue
      .slice(currentIdx + 1, currentIdx + 4)
      .filter((item) => !item.isRadio)
      .map((item) => item.trackId)

    if (nextSongIds.length > 0) {
      decisionService.prefetchDecisions(nextSongIds)
    }
  }, [playerState.queue, playerState.savedPlayIndex])

  const visible = authenticated && playerState.queue.length > 0
  const currentTrack = playerState.current || {}
  const currentTrackId = currentTrack.trackId
  const currentTrackUuid = currentTrack.uuid
  const isRadio = Boolean(currentTrack.isRadio)
  const remainingQueue = useMemo(
    () => getRemainingQueue(playerState),
    [playerState],
  )
  const classes = useStyle({
    isRadio,
    visible,
    enableCoverAnimation: config.enableCoverAnimation,
  })
  const showNotifications = useSelector(
    (state) => state.settings.notifications || false,
  )
  const streamingOverride = useSelector(
    (state) => state.settings?.streamingOverride || DEFAULT_STREAMING_OVERRIDE,
  )
  const gainInfo = useSelector((state) => state.replayGain)
  const [context, setContext] = useState(null)
  const [gainNode, setGainNode] = useState(null)
  const networkInfo = useMemo(() => {
    if (typeof navigator === 'undefined') {
      return null
    }
    return (
      navigator.connection ||
      navigator.mozConnection ||
      navigator.webkitConnection ||
      null
    )
  }, [])
  const canPreloadNextTrack = useMemo(() => {
    if (isMobilePlayer) {
      return false
    }
    if (!networkInfo) {
      return true
    }
    if (networkInfo.saveData) {
      return false
    }
    const slowTypes = ['slow-2g', '2g', '3g']
    return !slowTypes.includes(networkInfo.effectiveType)
  }, [isMobilePlayer, networkInfo])
  const streamQuery = useMemo(
    () => toStreamQuery(streamingOverride),
    [streamingOverride],
  )
  const overrideKey = useMemo(
    () => buildOverrideKey(streamingOverride),
    [streamingOverride],
  )
  const effectiveQueue = useMemo(
    () =>
      decorateQueueWithOverride(
        playerState.queue,
        overrideKey,
        streamQuery,
        subsonic.streamUrl,
      ),
    [playerState.queue, overrideKey, streamQuery],
  )
  const overrideKeyRef = useRef(overrideKey)
  const pendingResumeRef = useRef(null)

  useEffect(() => {
    if (
      context === null &&
      audioInstance &&
      config.enableReplayGain &&
      'AudioContext' in window &&
      (gainInfo.gainMode === 'album' || gainInfo.gainMode === 'track')
    ) {
      const ctx = new AudioContext()
      // we need this to support radios in firefox
      audioInstance.crossOrigin = 'anonymous'
      const source = ctx.createMediaElementSource(audioInstance)
      const gain = ctx.createGain()

      source.connect(gain)
      gain.connect(ctx.destination)

      setContext(ctx)
      setGainNode(gain)
    }
  }, [audioInstance, context, gainInfo.gainMode])

  useEffect(() => {
    if (gainNode) {
      const current = playerState.current || {}
      const song = current.song || {}

      const numericGain = calculateGain(gainInfo, song)
      gainNode.gain.setValueAtTime(numericGain, context.currentTime)
    }
  }, [audioInstance, context, gainNode, playerState, gainInfo])

  useEffect(() => {
    const handleBeforeUnload = (e) => {
      // Check there's a current track and is actually playing/not paused
      if (currentTrackUuid && audioInstance && !audioInstance.paused) {
        e.preventDefault()
        e.returnValue = '' // Chrome requires returnValue to be set
      }
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => window.removeEventListener('beforeunload', handleBeforeUnload)
  }, [audioInstance, currentTrackUuid])

  useEffect(() => {
    const previousKey = overrideKeyRef.current
    if (previousKey === overrideKey) {
      return
    }
    overrideKeyRef.current = overrideKey

    if (!currentTrackId || isRadio || !audioInstance) {
      pendingResumeRef.current = null
      return
    }

    const currentTime = Number(audioInstance.currentTime)
    pendingResumeRef.current = {
      trackId: currentTrackId,
      currentTime:
        Number.isFinite(currentTime) && currentTime > 0 ? currentTime : null,
      paused: Boolean(audioInstance.paused),
    }
  }, [audioInstance, overrideKey, currentTrackId, isRadio])

  const defaultOptions = useMemo(
    () => ({
      theme: playerTheme,
      bounds: 'body',
      playMode: playerState.mode,
      mode: 'full',
      loadAudioErrorPlayNext: false,
      autoPlayInitLoadPlayList: true,
      clearPriorAudioLists: false,
      showDestroy: true,
      showDownload: false,
      showLyric: true,
      showReload: false,
      toggleMode: !isDesktop,
      glassBg: false,
      showThemeSwitch: false,
      showMediaSession: true,
      restartCurrentOnPrev: true,
      quietUpdate: true,
      defaultPosition: {
        top: 300,
        left: 120,
      },
      volumeFade: { fadeIn: 200, fadeOut: 200 },
      renderAudioTitle: (audioInfo, isMobile) => (
        <AudioTitle
          audioInfo={audioInfo}
          gainInfo={gainInfo}
          isMobile={isMobile}
        />
      ),
      locale: locale(translate),
      sortableOptions: { delay: 200, delayOnTouchOnly: true },
    }),
    [gainInfo, isDesktop, playerTheme, translate, playerState.mode],
  )

  const options = useMemo(() => {
    const current = playerState.current || {}
    return {
      ...defaultOptions,
      audioLists: effectiveQueue.map((item) => item),
      playIndex: playerState.playIndex,
      autoPlay:
        playerState.autoPlay !== false &&
        (playerState.clear || playerState.playIndex === 0),
      clearPriorAudioLists: playerState.clear,
      extendsContent: (
        <PlayerToolbar id={current.trackId} isRadio={current.isRadio} />
      ),
      defaultVolume: isMobilePlayer ? 1 : playerState.volume,
      showMediaSession: !current.isRadio,
    }
  }, [playerState, defaultOptions, effectiveQueue, isMobilePlayer])

  const onAudioListsChange = useCallback(
    (_, audioLists, audioInfo) => dispatch(syncQueue(audioInfo, audioLists)),
    [dispatch],
  )

  const nextSong = useCallback(() => {
    const idx = effectiveQueue.findIndex(
      (item) => item.uuid === currentTrackUuid,
    )
    if (idx < 0) {
      return null
    }
    return effectiveQueue[idx + 1] || null
  }, [effectiveQueue, currentTrackUuid])

  const onAudioProgress = useCallback(
    (info) => {
      if (info.ended) {
        document.title = BRAND_NAME
      }

      const progress = (info.currentTime / info.duration) * 100
      if (isNaN(info.duration) || (progress < 50 && info.currentTime < 240)) {
        return
      }

      if (info.isRadio) {
        return
      }

      if (!preloaded) {
        if (!canPreloadNextTrack) {
          setPreload(true)
          return
        }
        const next = nextSong()
        if (next != null && !next.isRadio) {
          // Trigger decision pre-fetch (this also warms the cache)
          decisionService.prefetchDecisions([next.trackId])
        }
        setPreload(true)
        return
      }

      if (!scrobbled) {
        info.trackId && subsonic.scrobble(info.trackId, startTime)
        setScrobbled(true)
      }
    },
    [startTime, scrobbled, nextSong, preloaded, canPreloadNextTrack],
  )

  const onAudioVolumeChange = useCallback(
    // sqrt to compensate for the logarithmic volume
    (volume) => dispatch(setVolume(Math.sqrt(volume))),
    [dispatch],
  )

  const onAudioPlay = useCallback(
    (info) => {
      const pendingResume = pendingResumeRef.current
      if (pendingResume) {
        if (
          !info.isRadio &&
          info.trackId &&
          info.trackId === pendingResume.trackId &&
          audioInstance
        ) {
          const resumeAt = pendingResume.currentTime
          if (Number.isFinite(resumeAt) && resumeAt > 0) {
            try {
              audioInstance.currentTime = resumeAt
            } catch (e) {
              // Best-effort resume only; keep playback running on failure.
            }
          }
          if (pendingResume.paused) {
            setTimeout(() => {
              try {
                audioInstance.pause()
              } catch (e) {
                // Ignore pause errors in best-effort restore path.
              }
            }, 0)
          }
        }
        pendingResumeRef.current = null
      }

      // Do this to start the context; on chrome-based browsers, the context
      // will start paused since it is created prior to user interaction
      if (context && context.state !== 'running') {
        context.resume()
      }

      dispatch(currentPlaying(info))
      if (startTime === null) {
        setStartTime(Date.now())
      }
      if (info.duration) {
        const song = info.song
        document.title = `${song.title} - ${song.artist} - ${BRAND_NAME}`
        if (!info.isRadio) {
          const pos = startTime === null ? null : Math.floor(info.currentTime)
          subsonic.nowPlaying(info.trackId, pos)
        }
        setPreload(false)
        if (config.gaTrackingId) {
          ReactGA.event({
            category: 'Player',
            action: 'Play song',
            label: `${song.title} - ${song.artist}`,
          })
        }
        if (showNotifications) {
          sendNotification(
            song.title,
            `${song.artist} - ${song.album}`,
            info.cover,
          )
        }
      }
    },
    [audioInstance, context, dispatch, showNotifications, startTime],
  )

  const onAudioPlayTrackChange = useCallback(() => {
    if (scrobbled) {
      setScrobbled(false)
    }
    if (startTime !== null) {
      setStartTime(null)
    }
  }, [scrobbled, startTime])

  const onAudioPause = useCallback(
    (info) => dispatch(currentPlaying(info)),
    [dispatch],
  )

  useEffect(() => {
    if (!currentTrackId || isRadio) {
      return
    }
    dispatch(markAutoPlayTrackPlayed(currentTrackId))
  }, [currentTrackId, isRadio, dispatch])

  useEffect(() => {
    const queueHasPlayableTracks = playerState.queue.some(
      (item) => !item.isRadio && getQueueItemTrackId(item),
    )
    if (
      !autoplayState?.enabled ||
      autoplayState?.fetching ||
      !queueHasPlayableTracks ||
      isRadio
    ) {
      return
    }

    const bufferThreshold = Math.max(
      3,
      Math.floor((autoplayState.batchSize || 5) / 2),
    )
    if (remainingQueue > bufferThreshold) {
      return
    }

    refillAutoPlayQueue({
      autoplay: autoplayState,
      dataProvider,
      dispatch,
      player: playerState,
      source: 'player',
    })
  }, [
    autoplayState,
    dataProvider,
    dispatch,
    isRadio,
    playerState,
    remainingQueue,
  ])

  const onAudioEnded = useCallback(
    (currentPlayId, audioLists, info) => {
      setScrobbled(false)
      setStartTime(null)
      dispatch(currentPlaying(info))
      dataProvider
        .getOne('keepalive', { id: info.trackId })
        // eslint-disable-next-line no-console
        .catch((e) => console.log('Keepalive error:', e))
    },
    [dispatch, dataProvider],
  )

  const onCoverClick = useCallback((mode, audioLists, audioInfo) => {
    if (mode === 'full' && audioInfo?.song?.albumId) {
      window.location.href = `#/album/${audioInfo.song.albumId}/show`
    }
  }, [])

  const onAudioError = useCallback(
    (error, currentPlayId, audioLists, audioInfo) => {
      // Invalidate all cached decisions — token may be stale
      decisionService.invalidateAll()

      // Pre-fetch decisions for upcoming songs with fresh tokens
      const currentIdx = playerState.queue.findIndex(
        (item) => item.uuid === currentPlayId,
      )
      if (currentIdx >= 0) {
        const nextSongIds = playerState.queue
          .slice(currentIdx + 1, currentIdx + 4)
          .filter((item) => !item.isRadio)
          .map((item) => item.trackId)
        if (nextSongIds.length > 0) {
          decisionService.prefetchDecisions(nextSongIds)
        }
      }
    },
    [playerState.queue],
  )

  const onBeforeDestroy = useCallback(() => {
    return new Promise((resolve, reject) => {
      dispatch(clearQueue())
      reject()
    })
  }, [dispatch])

  if (!visible) {
    document.title = BRAND_NAME
  }

  const handlers = useMemo(
    () => keyHandlers(audioInstance, playerState),
    [audioInstance, playerState],
  )

  useEffect(() => {
    if (isMobilePlayer && audioInstance) {
      audioInstance.volume = 1
    }
  }, [isMobilePlayer, audioInstance])

  return (
    <ThemeProvider theme={createMuiTheme(theme)}>
      <ReactJkMusicPlayer
        {...options}
        className={classes.player}
        onAudioListsChange={onAudioListsChange}
        onAudioVolumeChange={onAudioVolumeChange}
        onAudioProgress={onAudioProgress}
        onAudioPlay={onAudioPlay}
        onAudioPlayTrackChange={onAudioPlayTrackChange}
        onAudioPause={onAudioPause}
        onPlayModeChange={(mode) => dispatch(setPlayMode(mode))}
        onAudioEnded={onAudioEnded}
        onCoverClick={onCoverClick}
        onAudioError={onAudioError}
        onBeforeDestroy={onBeforeDestroy}
        getAudioInstance={setAudioInstance}
      />
      <GlobalHotKeys handlers={handlers} keyMap={keyMap} allowChanges />
    </ThemeProvider>
  )
}

export { Player }
