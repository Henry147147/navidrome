import React from 'react'
import { Route } from 'react-router-dom'
import Personal from './personal/Personal'
import ExploreSuggestions from './explore/ExploreSuggestions'
import AutoPlayPage from './autoplay/AutoPlayPage'

const routes = [
  <Route
    exact
    path="/autoplay"
    render={() => <AutoPlayPage />}
    key={'autoplay'}
  />,
  <Route
    exact
    path="/explore"
    render={() => <ExploreSuggestions />}
    key={'explore'}
  />,
  <Route exact path="/personal" render={() => <Personal />} key={'personal'} />,
]

export default routes
