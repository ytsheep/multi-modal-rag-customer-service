const API_BASE = import.meta.env.VITE_API_BASE || ''

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `请求失败：${response.status}`)
  }
  return response.json()
}

export function sendChat(payload) {
  return request('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
}

export function uploadFile(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/api/upload', {
    method: 'POST',
    body: form
  })
}

export function listHistory(userId = 'default') {
  return request(`/api/history?user_id=${encodeURIComponent(userId)}`)
}

export function deleteHistory(sessionId, userId = 'default') {
  return request(`/api/history/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`, {
    method: 'DELETE'
  })
}
