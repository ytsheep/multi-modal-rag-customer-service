<script setup>
import { computed, onMounted, ref } from 'vue'
import { deleteHistory, listHistory, sendChat, uploadFile } from './api/client'

const userId = ref('default')
const sessionId = ref(createSessionId())
const question = ref('')
const messages = ref([])
const histories = ref([])
const citations = ref([])
const lastIntent = ref('')
const uploadState = ref({ loading: false, message: '' })
const chatLoading = ref(false)
const errorMessage = ref('')

const canSend = computed(() => question.value.trim().length > 0 && !chatLoading.value)

function createSessionId() {
  return `session-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`
}

async function refreshHistory() {
  histories.value = await listHistory(userId.value)
}

function newSession() {
  sessionId.value = createSessionId()
  messages.value = []
  citations.value = []
  lastIntent.value = ''
  errorMessage.value = ''
}

function openSession(session) {
  sessionId.value = session.session_id
  messages.value = session.messages.map((item) => ({
    role: item.role,
    content: item.content
  }))
  citations.value = []
  lastIntent.value = ''
}

async function removeSession(session) {
  await deleteHistory(session.session_id, userId.value)
  if (session.session_id === sessionId.value) {
    newSession()
  }
  await refreshHistory()
}

async function submitQuestion() {
  if (!canSend.value) return
  const text = question.value.trim()
  question.value = ''
  errorMessage.value = ''
  messages.value.push({ role: 'user', content: text })
  chatLoading.value = true
  try {
    const response = await sendChat({
      question: text,
      session_id: sessionId.value,
      user_id: userId.value
    })
    messages.value.push({ role: 'assistant', content: response.answer })
    citations.value = response.citations || []
    lastIntent.value = response.intent || ''
    await refreshHistory()
  } catch (error) {
    errorMessage.value = error.message
  } finally {
    chatLoading.value = false
  }
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  if (!file) return
  uploadState.value = { loading: true, message: `正在解析并入库：${file.name}` }
  errorMessage.value = ''
  try {
    const result = await uploadFile(file)
    uploadState.value = {
      loading: false,
      message: `已入库 ${result.file_name}，生成 ${result.chunks} 个资料片段`
    }
  } catch (error) {
    uploadState.value = { loading: false, message: '' }
    errorMessage.value = error.message
  } finally {
    event.target.value = ''
  }
}

function pageLabel(citation) {
  if (!citation.page_start && !citation.page_end) return '无页码'
  if (citation.page_start === citation.page_end) return `第 ${citation.page_start} 页`
  return `第 ${citation.page_start || '?'}-${citation.page_end || '?'} 页`
}

function scoreLabel(citation) {
  return Number(citation.score || 0).toFixed(2)
}

onMounted(refreshHistory)
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" role="img">
            <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8A2.5 2.5 0 0 1 17.5 16H9l-4.2 4.2A.5.5 0 0 1 4 19.85V5.5Z" />
          </svg>
        </div>
        <div>
          <h1>产品资料问答助手</h1>
          <p>ChromaDB · DashScope · LangChain</p>
        </div>
      </div>

      <button class="primary-action" type="button" @click="newSession">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg>
        新问答
      </button>

      <div class="history-list" aria-label="个人问答历史">
        <button
          v-for="session in histories"
          :key="session.session_id"
          class="history-item"
          :class="{ active: session.session_id === sessionId }"
          type="button"
          @click="openSession(session)"
        >
          <span>{{ session.title }}</span>
          <small>{{ session.updated_at }}</small>
        </button>
      </div>
    </aside>

    <main class="chat-area">
      <header class="topbar">
        <div>
          <strong>多模态RAG客服中台</strong>
          <span>意图分流 · 混合召回 · 重排序 Top 3 · 短期上下文</span>
        </div>
        <label class="user-field">
          <span>用户</span>
          <input v-model="userId" type="text" @change="refreshHistory" />
        </label>
      </header>

      <section class="message-pane" aria-live="polite">
        <div v-if="messages.length === 0" class="empty-state">
          <h2>上传产品手册后开始提问</h2>
          <p>支持普通 PDF、扫描件 PDF、Word、Markdown 和 TXT。参数、型号、单位会依据原文与页码回答。</p>
        </div>
        <article v-for="(message, index) in messages" :key="index" class="message" :class="message.role">
          <div class="avatar">{{ message.role === 'user' ? '问' : '答' }}</div>
          <p>{{ message.content }}</p>
        </article>
        <article v-if="chatLoading" class="message assistant">
          <div class="avatar">答</div>
          <p>正在检索产品资料并调用千问模型...</p>
        </article>
      </section>

      <form class="composer" @submit.prevent="submitQuestion">
        <label class="sr-only" for="question">输入问题</label>
        <textarea
          id="question"
          v-model="question"
          rows="3"
          placeholder="请输入产品问题，例如：产品A的额定输入电压是多少？请引用手册页码。"
          @keydown.ctrl.enter.prevent="submitQuestion"
        />
        <button type="submit" :disabled="!canSend">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 14-7-4 14-3-6-7-1Z" /></svg>
          发送
        </button>
      </form>

      <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
    </main>

    <aside class="inspector">
      <section class="panel">
        <h2>文件入库</h2>
        <label class="upload-box">
          <input type="file" accept=".pdf,.docx,.md,.markdown,.txt" @change="handleUpload" />
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m0 0 4 4m-4-4-4 4M5 20h14" /></svg>
          <span>{{ uploadState.loading ? '解析入库中' : '选择产品资料上传' }}</span>
        </label>
        <p v-if="uploadState.message" class="status-text">{{ uploadState.message }}</p>
      </section>

      <section class="panel">
        <h2>本次引用</h2>
        <div v-if="lastIntent" class="intent-badge">
          {{ lastIntent === 'direct_chat' ? '直接闲聊' : '产品资料问答' }}
        </div>
        <div v-if="citations.length === 0" class="muted">暂无引用片段</div>
        <article v-for="citation in citations" :key="citation.chunk_id" class="citation">
          <strong>{{ citation.file_name || '未知文件' }}</strong>
          <span>{{ pageLabel(citation) }} · score {{ scoreLabel(citation) }}</span>
          <p>{{ citation.content }}</p>
        </article>
      </section>

      <section class="panel">
        <h2>历史管理</h2>
        <button
          v-for="session in histories"
          :key="`delete-${session.session_id}`"
          class="ghost-action"
          type="button"
          @click="removeSession(session)"
        >
          删除：{{ session.title }}
        </button>
      </section>
    </aside>
  </div>
</template>
