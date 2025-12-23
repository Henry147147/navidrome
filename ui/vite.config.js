import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

const frontendPort = parseInt(process.env.PORT) || 4533
const backendPort = frontendPort + 100

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      manifest: manifest(),
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'sw.js',
      devOptions: {
        enabled: true,
      },
    }),
  ],
  server: {
    host: true,
    port: frontendPort,
    proxy: {
      '^/(auth|api|rest|backgrounds)/.*': 'http://localhost:' + backendPort,
    },
  },
  base: './',
  build: {
    outDir: 'build',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return
          }
          if (id.includes('/node_modules/navidrome-music-player/')) {
            return 'music-player'
          }
          if (id.includes('/node_modules/@material-ui/')) {
            return 'material-ui'
          }
          if (id.includes('/node_modules/ra-core/')) {
            return 'ra-core'
          }
          if (id.includes('/node_modules/ra-ui-materialui/')) {
            return 'ra-ui-materialui'
          }
          if (id.includes('/node_modules/ra-ui/')) {
            return 'ra-ui'
          }
          if (id.includes('/node_modules/react-admin/')) {
            return 'react-admin'
          }
          if (id.includes('/node_modules/ra-data-json-server/')) {
            return 'react-admin-data'
          }
          if (id.includes('/node_modules/ra-i18n-polyglot/')) {
            return 'react-admin-i18n'
          }
          if (id.includes('/node_modules/sortablejs/')) {
            return 'sortablejs'
          }
          if (id.includes('/node_modules/popper.js/')) {
            return 'popper'
          }
          if (id.includes('/node_modules/jss/')) {
            return 'jss'
          }
          if (id.includes('/node_modules/downshift/')) {
            return 'downshift'
          }
          if (id.includes('/node_modules/final-form/')) {
            return 'final-form'
          }
          if (id.includes('/node_modules/rc-slider/')) {
            return 'rc-slider'
          }
          if (id.includes('/node_modules/resize-observer-polyfill/')) {
            return 'resize-observer'
          }
          if (id.includes('/node_modules/lodash.isequalwith/')) {
            return 'lodash-isequalwith'
          }
          if (id.includes('/node_modules/lodash/')) {
            return 'lodash'
          }
          if (id.includes('/node_modules/react-dnd/')) {
            return 'react-dnd'
          }
          if (id.includes('/node_modules/react-icons/')) {
            return 'react-icons'
          }
          if (
            id.includes('/node_modules/react/') ||
            id.includes('/node_modules/react-dom/') ||
            id.includes('/node_modules/react-redux/') ||
            id.includes('/node_modules/react-router/') ||
            id.includes('/node_modules/react-router-dom/') ||
            id.includes('/node_modules/redux/') ||
            id.includes('/node_modules/connected-react-router/') ||
            id.includes('/node_modules/history/')
          ) {
            return 'react-core'
          }
          return 'vendor'
        },
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.js',
    css: true,
    reporters: ['verbose'],
    // reporters: ['default', 'hanging-process'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*'],
      exclude: [],
    },
  },
})

// PWA manifest
function manifest() {
  return {
    name: 'Hilberto Music',
    short_name: 'Hilberto Music',
    description:
      'Hilberto Music, an open source web-based music collection server and streamer',
    categories: ['music', 'entertainment'],
    display: 'standalone',
    start_url: './',
    background_color: 'white',
    theme_color: 'blue',
    icons: [
      {
        src: './android-chrome-192x192.png',
        sizes: '192x192',
        type: 'image/png',
      },
      {
        src: './android-chrome-512x512.png',
        sizes: '512x512',
        type: 'image/png',
      },
    ],
  }
}
