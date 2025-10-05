import React, { useMemo, useRef, useState } from 'react'
import {
  Button,
  Card,
  CardContent,
  CardActions,
  Divider,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Typography,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import CloudUploadIcon from '@material-ui/icons/CloudUpload'
import AttachFileIcon from '@material-ui/icons/AttachFile'
import ClearIcon from '@material-ui/icons/Clear'
import { Title, useNotify, useTranslate } from 'react-admin'
import { BRAND_NAME } from '../consts'
import { baseUrl } from '../utils'
import { applyAuthHeaders, updateAuthFromHeaders } from '../dataProvider'

const useStyles = makeStyles((theme) => ({
  root: {
    marginTop: theme.spacing(2),
    maxWidth: 720,
  },
  input: {
    display: 'none',
  },
  description: {
    marginBottom: theme.spacing(2),
  },
  emptyState: {
    color: theme.palette.text.secondary,
  },
  fileList: {
    marginTop: theme.spacing(1),
    marginBottom: theme.spacing(1),
  },
  actions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: theme.spacing(2),
  },
  itemProgress: {
    marginTop: theme.spacing(1),
  },
  itemProgressBar: {
    marginTop: theme.spacing(1),
  },
  itemProgressText: {
    marginTop: theme.spacing(0.5),
    display: 'block',
  },
  feedback: {
    marginTop: theme.spacing(2),
  },
}))

const MEGABYTE = 1024 * 1024
const AUDIO_EXTENSIONS = new Set([
  '.aac',
  '.aiff',
  '.aif',
  '.aifc',
  '.alac',
  '.ape',
  '.dsf',
  '.dff',
  '.flac',
  '.m4a',
  '.m4b',
  '.mka',
  '.mp3',
  '.mpc',
  '.ogg',
  '.opus',
  '.oga',
  '.spx',
  '.tta',
  '.wav',
  '.wv',
  '.wma',
])

const getFileExtension = (name) => {
  if (typeof name !== 'string') {
    return ''
  }
  const dotIndex = name.lastIndexOf('.')
  return dotIndex >= 0 ? name.slice(dotIndex).toLowerCase() : ''
}

const getFileBaseName = (name) => {
  if (typeof name !== 'string') {
    return ''
  }
  const dotIndex = name.lastIndexOf('.')
  return dotIndex >= 0 ? name.slice(0, dotIndex) : name
}

const isCueFile = (file) => getFileExtension(file?.name) === '.cue'

const isAudioFile = (file) => AUDIO_EXTENSIONS.has(getFileExtension(file?.name))

const getFileKey = (file) => `${file.name}-${file.lastModified}`

const createGroup = (primary, companions = []) => {
  const files = [primary, ...companions]
  const fileKeys = files.map(getFileKey)
  return {
    id: fileKeys.join('|'),
    primary,
    companions,
    files,
    fileKeys,
  }
}

const buildUploadGroups = (fileList) => {
  const baseMap = new Map()
  const groupByFileKey = new Map()

  fileList.forEach((file) => {
    if (!file) {
      return
    }
    if (isCueFile(file) || isAudioFile(file)) {
      const baseKey = getFileBaseName(file.name).toLowerCase()
      const entry = baseMap.get(baseKey) || { audio: [], cues: [] }
      if (isCueFile(file)) {
        entry.cues.push(file)
      } else {
        entry.audio.push(file)
      }
      baseMap.set(baseKey, entry)
    }
  })

  baseMap.forEach((entry) => {
    const remainingCues = [...entry.cues]

    entry.audio.forEach((audioFile) => {
      const companions = remainingCues.length ? [remainingCues.shift()] : []
      const group = createGroup(audioFile, companions)
      group.files.forEach((file) => {
        groupByFileKey.set(getFileKey(file), group)
      })
    })

    remainingCues.forEach((cueFile) => {
      const group = createGroup(cueFile)
      group.files.forEach((file) => {
        groupByFileKey.set(getFileKey(file), group)
      })
    })
  })

  const orderedGroups = []
  const seenGroupIds = new Set()

  fileList.forEach((file) => {
    if (!file) {
      return
    }
    const fileKey = getFileKey(file)
    let group = groupByFileKey.get(fileKey)
    if (!group) {
      group = createGroup(file)
      group.files.forEach((member) => {
        groupByFileKey.set(getFileKey(member), group)
      })
    }
    if (!seenGroupIds.has(group.id)) {
      orderedGroups.push(group)
      seenGroupIds.add(group.id)
    }
  })

  return orderedGroups
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  )
  const value = bytes / Math.pow(1024, index)
  const precision = value >= 10 || index === 0 ? 0 : 1
  return `${value.toFixed(precision)} ${units[index]}`
}

const formatMegabytes = (bytes, precision) => {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0.0'
  }
  const value = bytes / MEGABYTE
  const decimals =
    typeof precision === 'number' ? precision : value >= 10 ? 1 : 2
  return value.toFixed(decimals)
}

const formatDuration = (seconds) => {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return null
  }
  const totalSeconds = Math.ceil(seconds)
  const minutes = Math.floor(totalSeconds / 60)
  const remainingSeconds = totalSeconds % 60
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`
  }
  return `${remainingSeconds}s`
}

const uploadFileGroup = (files, onProgress) => {
  const url = baseUrl('/api/upload')

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url)

    const baseHeaders = applyAuthHeaders(
      typeof Headers !== 'undefined' ? new Headers() : {},
    )
    if (typeof Headers !== 'undefined' && baseHeaders instanceof Headers) {
      baseHeaders.forEach((value, key) => {
        xhr.setRequestHeader(key, value)
      })
    } else {
      Object.entries(baseHeaders).forEach(([key, value]) => {
        xhr.setRequestHeader(key, value)
      })
    }

    const formData = new FormData()
    const fileArray = Array.isArray(files) ? files : [files]
    fileArray.forEach((file) => {
      formData.append('file', file, file.name)
    })

    const startTime = Date.now()
    const totalBytes = fileArray.reduce((sum, current) => sum + current.size, 0)

    if (onProgress) {
      onProgress({
        loaded: 0,
        total: totalBytes,
        lengthComputable: totalBytes > 0,
        speed: 0,
        eta: null,
      })
    }

    xhr.upload.addEventListener('progress', (event) => {
      if (!onProgress) {
        return
      }
      const elapsedSeconds = (Date.now() - startTime) / 1000
      const speed = elapsedSeconds > 0 ? event.loaded / elapsedSeconds : 0
      const eta =
        event.lengthComputable && speed > 0
          ? (event.total - event.loaded) / speed
          : null
      onProgress({
        loaded: event.loaded,
        total: event.lengthComputable ? event.total : totalBytes,
        lengthComputable: event.lengthComputable,
        speed,
        eta,
      })
    })

    xhr.addEventListener('error', () => {
      reject(new Error('Network error during upload'))
    })

    xhr.addEventListener('abort', () => {
      reject(new Error('Upload aborted'))
    })

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        updateAuthFromHeaders({
          get: (name) => xhr.getResponseHeader(name),
        })
        let body = null
        try {
          body = xhr.responseText ? JSON.parse(xhr.responseText) : null
        } catch (error) {
          body = null
        }
        resolve(body)
        return
      }

      let errorBody = null
      try {
        errorBody = xhr.responseText ? JSON.parse(xhr.responseText) : null
      } catch (error) {
        errorBody = null
      }

      const message =
        (errorBody && errorBody.message) || xhr.statusText || 'Upload failed'
      const err = new Error(message)
      err.status = xhr.status
      err.body = errorBody
      reject(err)
    })

    xhr.send(formData)
  })
}

const UploadPage = () => {
  const translate = useTranslate()
  const notify = useNotify()
  const classes = useStyles()
  const inputRef = useRef(null)
  const [files, setFiles] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [feedbackMessage, setFeedbackMessage] = useState('')
  const [currentProgress, setCurrentProgress] = useState(null)

  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + file.size, 0),
    [files],
  )

  const handleFileSelection = (event) => {
    const selectedFiles = Array.from(event.target.files || [])
    setFiles(selectedFiles)
    setFeedbackMessage('')
  }

  const clearSelection = () => {
    setFiles([])
    setFeedbackMessage('')
    setCurrentProgress(null)
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  const handleUpload = async () => {
    if (!files.length || isUploading) {
      if (!isUploading) {
        const message = translate('upload.notifications.noFiles')
        setFeedbackMessage(message)
        notify(message, 'warning')
      }
      return
    }

    setIsUploading(true)
    setFeedbackMessage('')

    const queue = buildUploadGroups(files)
    let uploadedCount = 0

    while (queue.length) {
      const currentGroup = queue[0]
      const groupTotalSize = currentGroup.files.reduce(
        (sum, file) => sum + file.size,
        0,
      )

      setCurrentProgress({
        id: currentGroup.id,
        fileKeys: currentGroup.fileKeys,
        displayName: currentGroup.primary.name,
        loaded: 0,
        total: groupTotalSize,
        lengthComputable: groupTotalSize > 0,
        speed: 0,
        eta: null,
      })

      try {
        await uploadFileGroup(currentGroup.files, (progress) => {
          setCurrentProgress((prev) => {
            if (!prev || prev.id !== currentGroup.id) {
              return prev
            }
            const nextTotal = progress.lengthComputable
              ? progress.total
              : prev.total
            return {
              ...prev,
              ...progress,
              total: nextTotal,
            }
          })
        })
        uploadedCount += currentGroup.files.length
        queue.shift()
        const completedKeys = new Set(currentGroup.fileKeys)
        setFiles((prev) =>
          prev.filter((file) => !completedKeys.has(getFileKey(file))),
        )
      } catch (error) {
        const defaultMessage = translate('upload.notifications.error')
        const message =
          (error?.body && error.body.message) || error?.message || defaultMessage
        setFeedbackMessage(message)
        notify(defaultMessage, 'warning')
        break
      }
    }

    if (queue.length === 0 && uploadedCount > 0) {
      notify(
        translate('upload.notifications.success', {
          smart_count: uploadedCount,
        }),
        'info',
      )
      if (inputRef.current) {
        inputRef.current.value = ''
      }
    }

    setIsUploading(false)
    setCurrentProgress(null)
  }

  return (
    <Card className={classes.root}>
      <Title title={`${BRAND_NAME} - ${translate('menu.upload.name')}`} />
      <CardContent>
        <Typography variant="h5" gutterBottom>
          {translate('upload.title')}
        </Typography>
        <Typography variant="body1" className={classes.description}>
          {translate('upload.description')}
        </Typography>

        <input
          ref={inputRef}
          className={classes.input}
          id="upload-files-input"
          type="file"
          multiple
          onChange={handleFileSelection}
        />
        <label htmlFor="upload-files-input">
          <Button
            startIcon={<AttachFileIcon />}
            variant="outlined"
            color="default"
            component="span"
            disabled={isUploading}
          >
            {translate('upload.actions.select')}
          </Button>
        </label>

        <div className={classes.fileList}>
          {files.length ? (
            <>
              <Typography variant="subtitle1">
                {translate('upload.selectedCount', { smart_count: files.length })}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {translate('upload.totalSize', { size: formatFileSize(totalSize) })}
              </Typography>
              <List dense>
                {files.map((file) => {
                  const fileKey = getFileKey(file)
                  const isCurrent =
                    isUploading && currentProgress?.fileKeys?.includes(fileKey)
                  const progress = isCurrent ? currentProgress : null
                  const progressTotalBytes = progress?.lengthComputable
                    ? progress.total
                    : file.size
                  const progressPercent = progress?.lengthComputable
                    ? (progress.loaded / Math.max(progressTotalBytes, 1)) * 100
                    : 0
                  const speedText = progress && progress.speed > 0
                    ? translate('upload.progress.speed', {
                        value: formatMegabytes(progress.speed, 2),
                      })
                    : translate('upload.progress.calculating')
                  const etaDuration = formatDuration(progress?.eta)
                  const etaText = etaDuration
                    ? translate('upload.progress.eta', { time: etaDuration })
                    : translate('upload.progress.calculating')
                  const detailText = progress
                    ? translate('upload.progress.detail', {
                        loaded: formatMegabytes(progress.loaded, 2),
                        total: formatMegabytes(progressTotalBytes, 2),
                        speed: speedText,
                        eta: etaText,
                      })
                    : null

                  return (
                    <ListItem key={fileKey} divider>
                      <ListItemText
                        primary={file.name}
                        secondary={
                          <>
                            {`${formatFileSize(file.size)} • ${new Date(
                              file.lastModified,
                            ).toLocaleString()}`}
                            {progress && (
                              <div className={classes.itemProgress}>
                                <LinearProgress
                                  className={classes.itemProgressBar}
                                  variant={
                                    progress.lengthComputable
                                      ? 'determinate'
                                      : 'indeterminate'
                                  }
                                  value={
                                    progress.lengthComputable
                                      ? Math.min(progressPercent, 100)
                                      : undefined
                                  }
                                />
                                <Typography
                                  variant="caption"
                                  color="textSecondary"
                                  className={classes.itemProgressText}
                                >
                                  {translate('upload.progress.uploading', {
                                    name: file.name,
                                  })}
                                </Typography>
                                <Typography
                                  variant="caption"
                                  color="textSecondary"
                                  className={classes.itemProgressText}
                                >
                                  {detailText}
                                </Typography>
                              </div>
                            )}
                          </>
                        }
                      />
                    </ListItem>
                  )
                })}
              </List>
            </>
          ) : (
            <Typography variant="body2" className={classes.emptyState}>
              {translate('upload.emptySelection')}
            </Typography>
          )}
        </div>
        {feedbackMessage && (
          <Typography
            variant="body2"
            color="error"
            className={classes.feedback}
            role="alert"
          >
            {feedbackMessage}
          </Typography>
        )}
      </CardContent>
      <Divider />
      <CardActions className={classes.actions}>
        <Button
          startIcon={<CloudUploadIcon />}
          variant="contained"
          color="primary"
          onClick={handleUpload}
          disabled={!files.length || isUploading}
        >
          {translate('upload.actions.upload')}
        </Button>
        <Button
          startIcon={<ClearIcon />}
          onClick={clearSelection}
          disabled={!files.length || isUploading}
        >
          {translate('upload.actions.clear')}
        </Button>
      </CardActions>
    </Card>
  )
}

export default UploadPage
