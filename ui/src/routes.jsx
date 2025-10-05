import React from 'react'
import { Route } from 'react-router-dom'
import Personal from './personal/Personal'
import ExploreSuggestions from './explore/ExploreSuggestions'
import UploadPage from './upload/UploadPage'

const routes = [
  <Route exact path="/upload" render={() => <UploadPage />} key={'upload'} />,
  <Route
    exact
    path="/explore"
    render={() => <ExploreSuggestions />}
    key={'explore'}
  />,
  <Route exact path="/personal" render={() => <Personal />} key={'personal'} />,
]

export default routes
