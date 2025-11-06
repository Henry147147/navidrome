import React from 'react'
import { Title, useTranslate } from 'react-admin'
import { Box, Typography } from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import { BRAND_NAME } from '../consts'
import BatchEmbeddingPanel from '../settings/BatchEmbeddingPanel'

const useStyles = makeStyles((theme) => ({
  root: {
    marginTop: theme.spacing(2),
  },
  section: {
    marginBottom: theme.spacing(4),
  },
  header: {
    marginBottom: theme.spacing(2),
  },
}))

const AdminSettings = () => {
  const translate = useTranslate()
  const classes = useStyles()

  // Check if user is admin (from localStorage)
  const role = localStorage.getItem('role')
  const isAdmin = role === 'admin'

  if (!isAdmin) {
    return (
      <Box className={classes.root}>
        <Title title={`${BRAND_NAME} - ${translate('menu.admin.name', { _: 'Admin' })}`} />
        <Typography variant="h6" color="error">
          {translate('ra.auth.auth_check_error', { _: 'Unauthorized' })}
        </Typography>
        <Typography variant="body2" color="textSecondary">
          {translate('pages.admin.requiresAdmin', {
            _: 'This page requires administrator privileges.',
          })}
        </Typography>
      </Box>
    )
  }

  return (
    <Box className={classes.root}>
      <Title title={`${BRAND_NAME} - ${translate('menu.admin.name', { _: 'Admin Settings' })}`} />

      <Box className={classes.section}>
        <Box className={classes.header}>
          <Typography variant="h4" gutterBottom>
            {translate('pages.admin.title', {
              _: 'Administration',
            })}
          </Typography>
          <Typography variant="body1" color="textSecondary">
            {translate('pages.admin.description', {
              _: 'System administration and maintenance tools.',
            })}
          </Typography>
        </Box>

        <BatchEmbeddingPanel />
      </Box>
    </Box>
  )
}

export default AdminSettings
