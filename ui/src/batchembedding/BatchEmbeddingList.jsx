import React from 'react'
import { Title, useTranslate } from 'react-admin'
import { Card, CardContent } from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import { BRAND_NAME } from '../consts'
import BatchEmbeddingPanel from '../settings/BatchEmbeddingPanel'

const useStyles = makeStyles((theme) => ({
  root: {
    marginTop: theme.spacing(2),
  },
}))

const BatchEmbeddingList = () => {
  const translate = useTranslate()
  const classes = useStyles()

  return (
    <div className={classes.root}>
      <Title
        title={`${BRAND_NAME} - ${translate('resources.batchembedding.name', {
          _: 'Batch Re-embedding',
        })}`}
      />
      <Card>
        <CardContent>
          <BatchEmbeddingPanel />
        </CardContent>
      </Card>
    </div>
  )
}

export default BatchEmbeddingList
