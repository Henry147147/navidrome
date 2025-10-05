import { fetchUtils } from 'react-admin'
import { v4 as uuidv4 } from 'uuid'
import { baseUrl } from '../utils'
import config from '../config'
import { jwtDecode } from 'jwt-decode'
import { removeHomeCache } from '../utils/removeHomeCache'

export const customAuthorizationHeader = 'X-ND-Authorization'
export const clientUniqueIdHeader = 'X-ND-Client-Unique-Id'
const clientUniqueId = uuidv4()

export const applyAuthHeaders = (incomingHeaders) => {
  const hasHeadersConstructor = typeof Headers !== 'undefined'
  let headers = incomingHeaders
  if (hasHeadersConstructor) {
    headers =
      headers instanceof Headers ? headers : new Headers(headers || {})
    if (!headers.has('Accept')) {
      headers.set('Accept', 'application/json')
    }
    headers.set(clientUniqueIdHeader, clientUniqueId)
    const token = localStorage.getItem('token')
    if (token) {
      headers.set(customAuthorizationHeader, `Bearer ${token}`)
    }
    return headers
  }

  headers = { ...(headers || {}) }
  if (!headers.Accept) {
    headers.Accept = 'application/json'
  }
  headers[clientUniqueIdHeader] = clientUniqueId
  const token = localStorage.getItem('token')
  if (token) {
    headers[customAuthorizationHeader] = `Bearer ${token}`
  }
  return headers
}

export const updateAuthFromHeaders = (headers) => {
  if (!headers || typeof headers.get !== 'function') {
    return
  }
  const token = headers.get(customAuthorizationHeader)
  if (token) {
    try {
      const decoded = jwtDecode(token)
      localStorage.setItem('token', token)
      localStorage.setItem('userId', decoded.uid)
      // Avoid going to create admin dialog after logout/login without a refresh
      config.firstTime = false
      removeHomeCache()
    } catch (error) {
      // Ignore malformed tokens and keep previous auth state
    }
  }
}

const httpClient = (url, options = {}) => {
  url = baseUrl(url)
  const headers = applyAuthHeaders(options.headers || {})
  options.headers = headers
  return fetchUtils.fetchJson(url, options).then((response) => {
    updateAuthFromHeaders(response.headers)
    return response
  })
}

export default httpClient
