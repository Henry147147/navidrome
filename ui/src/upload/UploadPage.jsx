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
import { httpClient } from '../dataProvider'

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
  progress: {
    marginTop: theme.spacing(2),
  },
  feedback: {
    marginTop: theme.spacing(2),
  },
}))

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

const UploadPage = () => {
  const translate = useTranslate()
  const notify = useNotify()
  const classes = useStyles()
  const inputRef = useRef(null)
  const [files, setFiles] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [feedbackMessage, setFeedbackMessage] = useState('')

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
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  const handleUpload = async () => {
    if (!files.length) {
      const message = translate('upload.notifications.noFiles')
      setFeedbackMessage(message)
      notify(message, 'warning')
      return
    }

    setIsUploading(true)
    setFeedbackMessage('')

    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file, file.name)
    })

    try {
      const headers = new Headers({ Accept: 'application/json' })
      const response = await httpClient('/api/upload', {
        method: 'POST',
        body: formData,
        headers,
      })

      const uploadedCount = response?.json?.files?.length || files.length
      notify(
        translate('upload.notifications.success', {
          smart_count: uploadedCount,
        }),
        'info',
      )
      clearSelection()
    } catch (error) {
      const defaultMessage = translate('upload.notifications.error')
      const message =
        (error?.body && error.body.message) || error?.message || defaultMessage
      setFeedbackMessage(message)
      notify(defaultMessage, 'warning')
    } finally {
      setIsUploading(false)
    }
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
                {files.map((file) => (
                  <ListItem key={`${file.name}-${file.lastModified}`} divider>
                    <ListItemText
                      primary={file.name}
                      secondary={`${formatFileSize(file.size)} • ${new Date(
                        file.lastModified,
                      ).toLocaleString()}`}
                    />
                  </ListItem>
                ))}
              </List>
            </>
          ) : (
            <Typography variant="body2" className={classes.emptyState}>
              {translate('upload.emptySelection')}
            </Typography>
          )}
        </div>

        {isUploading && <LinearProgress className={classes.progress} />}

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
