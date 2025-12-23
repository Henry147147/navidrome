import { lazy, Suspense } from 'react'
import config, { shareInfo } from './config'

const AdminApp = lazy(() => import('./AdminApp'))
const SharePlayer = lazy(() => import('./share/SharePlayer'))

const App = () => {
  if (config.enableSharing && shareInfo) {
    return (
      <Suspense fallback={null}>
        <SharePlayer />
      </Suspense>
    )
  }
  return (
    <Suspense fallback={null}>
      <AdminApp />
    </Suspense>
  )
}

export default App
