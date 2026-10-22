import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import axios from 'axios'
import { io } from 'socket.io-client'
import { useComfort } from './ComfortContext.jsx'

const API         = 'http://localhost:5000'
const LOCK_SECS   = 1.0    // seconds a word must be stable before locking in
const CONF_THRESH = 0.35   // align with backend CONFIDENCE
const LANGUAGES   = ['english', 'hindi', 'marathi', 'spanish', 'french', 'arabic']

// ── Send canvas resolution (enough for MediaPipe keypoints, smaller = faster) ─
const SEND_W = 640
const SEND_H = 480
const JPEG_Q = 0.5         // JPEG quality for network transfer

// ── Simple sentence builder ───────────────────────────────────────────────────
class SentenceBuilder {
  constructor() { this.reset() }
  reset() { this.words = []; this.candidate = null; this.candSince = null; this.lastWord = null }
  update(word, conf) {
    const now = Date.now()
    if (conf < CONF_THRESH) { this.candidate = null; return null }
    if (word !== this.candidate) { this.candidate = word; this.candSince = now }
    if ((now - this.candSince) / 1000 >= LOCK_SECS && word !== this.lastWord) {
      this.words.push(word)
      this.lastWord = word
      this.candidate = null
      return word
    }
    return null
  }
  undo() { if (this.words.length) { const w = this.words.pop(); this.lastWord = this.words.at(-1) || null; return w } return null }
  lockProgress() {
    if (!this.candidate || !this.candSince) return 0
    return Math.min((Date.now() - this.candSince) / (LOCK_SECS * 1000), 1)
  }
}

export default function Direction1({ onBack }) {
  const { prefs } = useComfort()
  const videoRef      = useRef(null)
  const drawCanvasRef = useRef(null)   // visible overlay canvas
  const drawCtxRef    = useRef(null)   // cached 2D context
  const sendCanvasRef = useRef(null)   // off-screen capture canvas (hidden)
  const sendCtxRef    = useRef(null)   // cached send context
  const streamRef     = useRef(null)
  const socketRef     = useRef(null)
  const sbRef         = useRef(new SentenceBuilder())

  // Refs shared with rAF landmark loop — no re-render cost
  const landmarksRef    = useRef({ pose: [], lh: [], rh: [] })
  const capturingRef    = useRef(false)
  const pendingRef      = useRef(false)   // frame in-flight over WS?
  const rafLandmarkRef  = useRef(null)

  const [capturing,    setCapturing]    = useState(false)
  const [predictions,  setPredictions]  = useState([])
  const [lockProg,     setLockProg]     = useState(0)
  const [words,        setWords]        = useState([])
  const [sentence,     setSentence]     = useState('')
  const [translations, setTranslations] = useState({})
  const [audio,        setAudio]        = useState(null)
  const [activeLang,   setActiveLang]   = useState('english')
  const [loading,      setLoading]      = useState(false)
  const [camError,     setCamError]     = useState(null)
  const [wsStatus,     setWsStatus]     = useState('disconnected') // 'connected'|'disconnected'
  const [modelLabel,   setModelLabel]   = useState('')           // e.g. 'v1 · baseline'

  // Ask the backend which model (v1 / v2) it is serving
  useEffect(() => {
    axios.get(`${API}/`)
      .then(res => setModelLabel(res.data.model_label || res.data.model || ''))
      .catch(() => {})
  }, [])

  // ── WebSocket connection ────────────────────────────────────────────────────
  useEffect(() => {
    const socket = io(API, {
      transports: ['websocket'],       // skip polling upgrade; go straight to WS
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
    })
    socketRef.current = socket

    socket.on('connect',    () => setWsStatus('connected'))
    socket.on('disconnect', () => setWsStatus('disconnected'))

    socket.on('prediction', (data) => {
      pendingRef.current = false       // free to send next frame

      if (data.error) return

      // Store landmarks in ref (read by 60-fps rAF loop, no re-render)
      if (data.landmarks) {
        landmarksRef.current = data.landmarks
      }

      // Update React state for prediction display
      const preds = data.predictions || []
      if (preds.length > 0) {
        setPredictions(preds)
        const newWord = sbRef.current.update(preds[0].word, preds[0].conf)
        if (newWord) setWords([...sbRef.current.words])
      }
      setLockProg(sbRef.current.lockProgress())

      // Immediately send next frame if still capturing (ping-pong)
      if (capturingRef.current) {
        sendNextFrame()
      }
    })

    return () => socket.disconnect()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Start webcam ────────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 30 } }
      })
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
    } catch {
      setCamError('Camera access denied. Please allow camera permission.')
    }
  }, [])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
  }, [])

  // ── 60-fps landmark render loop (decoupled from network) ───────────────────
  const startLandmarkLoop = useCallback(() => {
    const loop = () => {
      const canvas = drawCanvasRef.current
      const video  = videoRef.current
      if (!canvas || !video || !video.videoWidth) {
        rafLandmarkRef.current = requestAnimationFrame(loop)
        return
      }

      // Cache context once
      if (!drawCtxRef.current) {
        drawCtxRef.current = canvas.getContext('2d')
      }

      const vw = video.clientWidth
      const vh = video.clientHeight

      // Only resize canvas if dimensions actually changed
      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width  = vw
        canvas.height = vh
      }

      const ctx = drawCtxRef.current
      ctx.clearRect(0, 0, vw, vh)

      const lm = landmarksRef.current
      if (lm) {
        const dot = (pts, color, size) => {
          if (!pts?.length) return
          ctx.fillStyle = color
          for (const pt of pts) {
            ctx.beginPath()
            ctx.arc(pt.x * vw, pt.y * vh, size, 0, 2 * Math.PI)
            ctx.fill()
          }
        }
        dot(lm.pose, '#B8C4A8', 3)
        dot(lm.lh,   '#5BADDE', 2.5)
        dot(lm.rh,   '#DE8B5B', 2.5)
      }

      rafLandmarkRef.current = requestAnimationFrame(loop)
    }
    rafLandmarkRef.current = requestAnimationFrame(loop)
  }, [])

  const stopLandmarkLoop = useCallback(() => {
    if (rafLandmarkRef.current) cancelAnimationFrame(rafLandmarkRef.current)
    rafLandmarkRef.current = null
    // Clear overlay
    const canvas = drawCanvasRef.current
    if (canvas && drawCtxRef.current) {
      drawCtxRef.current.clearRect(0, 0, canvas.width, canvas.height)
    }
    landmarksRef.current = { pose: [], lh: [], rh: [] }
  }, [])

  // ── Frame capture + send (ping-pong via WebSocket) ─────────────────────────
  const sendNextFrame = useCallback(() => {
    if (!capturingRef.current || pendingRef.current) return
    const video  = videoRef.current
    const canvas = sendCanvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    // Cache ctx and fix canvas size once
    if (!sendCtxRef.current) {
      sendCtxRef.current = canvas.getContext('2d')
      canvas.width  = SEND_W
      canvas.height = SEND_H
    }

    sendCtxRef.current.drawImage(video, 0, 0, SEND_W, SEND_H)
    const b64 = canvas.toDataURL('image/jpeg', JPEG_Q).split(',')[1]

    pendingRef.current = true
    socketRef.current?.emit('frame', { frame_b64: b64 })
  }, [])

  // ── Capture control ─────────────────────────────────────────────────────────
  const startCapture = useCallback(() => {
    sbRef.current.reset()
    setWords([])
    setSentence('')
    setTranslations({})
    setAudio(null)
    setPredictions([])
    capturingRef.current = true
    setCapturing(true)
    pendingRef.current = false
    startLandmarkLoop()
    // Kick off the first frame immediately
    sendNextFrame()
  }, [sendNextFrame, startLandmarkLoop])

  const stopCapture = useCallback(() => {
    capturingRef.current = false
    pendingRef.current   = false
    setCapturing(false)
    stopLandmarkLoop()
  }, [stopLandmarkLoop])

  // ── Action Handlers ─────────────────────────────────────────────────────────
  const completeSentence = useCallback(async () => {
    if (words.length === 0) return
    stopCapture()
    setLoading(true)
    try {
      const res = await axios.post(`${API}/complete-sentence`, { words })
      setSentence(res.data.sentence || '')
      setTranslations(res.data.translations || {})
      if (res.data.audio_b64) {
        const blob = new Blob(
          [Uint8Array.from(atob(res.data.audio_b64), c => c.charCodeAt(0))],
          { type: 'audio/mpeg' }
        )
        setAudio(URL.createObjectURL(blob))
      }
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }, [words, stopCapture])

  const undo  = useCallback(() => { sbRef.current.undo(); setWords([...sbRef.current.words]) }, [])
  const clear = useCallback(() => {
    sbRef.current.reset()
    setWords([]); setSentence(''); setTranslations({}); setAudio(null); setPredictions([])
  }, [])

  // ── Keyboard shortcuts ──────────────────────────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      switch (e.key.toLowerCase()) {
        case 'enter':     e.preventDefault(); completeSentence(); break
        case 'backspace': e.preventDefault(); undo(); break
        case 'c':         e.preventDefault(); clear(); break
        case 'q':         e.preventDefault(); onBack(); break
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [completeSentence, undo, clear, onBack])

  // ── Mount / unmount ─────────────────────────────────────────────────────────
  useEffect(() => {
    startCamera()
    return () => { stopCamera(); stopCapture() }
  }, [startCamera, stopCamera, stopCapture])

  const top1     = predictions.length > 0 ? predictions[0] : null
  const subPreds = predictions.slice(1, 5)
  const displayText = activeLang === 'english' ? sentence : (translations[activeLang] || '')

  const liveAnnouncement = useMemo(() => {
    const parts = []
    if (loading) parts.push('Working on your sentence.')
    if (capturing && top1) parts.push(`Latest guess: ${top1.word}, ${Math.round(top1.conf * 100)} percent.`)
    if (words.length > 0) parts.push(`Words collected: ${words.join(', ')}.`)
    if (sentence) parts.push(`Sentence ready: ${sentence}`)
    return parts.join(' ') || 'Gesture to language. Allow the camera, then start capture when ready.'
  }, [capturing, top1, words, sentence, loading])

  return (
    <>
      <a href="#gesture-main" className="skip-link">Skip to main content</a>
      <main id="gesture-main" className="screen" tabIndex={-1}>
        <div className="sr-only" aria-live="polite" aria-atomic="true">{liveAnnouncement}</div>

        <header className="screen-header">
          <button className="back-btn" onClick={onBack} title="Back (Q)">←</button>
          <h1 className="screen-title">Gesture to language</h1>
          <span className="screen-tag">
            Camera · live interpretation{modelLabel && ` · model ${modelLabel}`}
            <span
              title={`WebSocket: ${wsStatus}`}
              style={{
                marginLeft: '0.5rem',
                display: 'inline-block',
                width: 8, height: 8,
                borderRadius: '50%',
                background: wsStatus === 'connected' ? '#4ade80' : '#f87171',
                verticalAlign: 'middle',
              }}
            />
          </span>
        </header>

        <div className="screen-body">

          {/* ── Webcam HUD ── */}
          <div className="panel" style={{ padding: 0 }}>
            <div className="hud-inner-container">
              {camError ? (
                <div className="webcam-placeholder">🚫 {camError}</div>
              ) : (
                <video ref={videoRef} className="hud-video" autoPlay playsInline muted />
              )}

              <canvas ref={drawCanvasRef} className="hud-draw-canvas" />
              <canvas ref={sendCanvasRef} style={{ display: 'none' }} />

              {/* Word HUD Overlays */}
              {capturing && top1 && (
                <>
                  <div className="hud-top-left">
                    <span className="hud-main-word">{top1.word}</span>
                    <span className="hud-main-conf">{(top1.conf * 100).toFixed(0)}%</span>
                  </div>
                  <div className="hud-top-right">
                    {subPreds.map((p, i) => (
                      <div key={i} className="hud-sub-pred">
                        {i + 2}. {p.word} {(p.conf * 100).toFixed(0)}%
                      </div>
                    ))}
                  </div>
                  <div className="hud-progress-container">
                    <div className="hud-progress-bar" style={{ width: `${lockProg * 100}%` }} />
                  </div>
                </>
              )}
            </div>
          </div>

          {/* ── Capture Controls ── */}
          <div className="panel">
            <p className="panel-title">Capture &amp; Controls</p>
            <div className="capture-controls">
              {!capturing ? (
                <button
                  className="btn btn-primary btn-lg"
                  onClick={startCapture}
                  disabled={!!camError || wsStatus !== 'connected'}
                >
                  {wsStatus !== 'connected' ? '⏳ Connecting…' : '▶ Start Capturing'}
                </button>
              ) : (
                <button className="btn btn-danger btn-lg" onClick={stopCapture}>
                  ⏹ Stop
                </button>
              )}
              <button className="btn btn-ghost" onClick={undo} disabled={words.length === 0}>
                ↩ Undo
              </button>
              <button className="btn btn-ghost" onClick={clear} disabled={words.length === 0 && !sentence}>
                🗑 Clear
              </button>
              <button
                className="btn btn-green btn-lg"
                onClick={completeSentence}
                disabled={words.length === 0 || loading}
              >
                {loading ? <><span className="spinner" /> Processing…</> : '✔ Complete Sentence'}
              </button>
            </div>

            <div className="hud-keyboard-rules">
              Keyboard Shortcuts: [ENTER] = Complete Sentence • [Backspace] = Delete Last Word • [C] = Clear Screen • [Q] = Quit
            </div>
          </div>

          {/* ── Detected Words Panel ── */}
          <div className="panel">
            <p className="panel-title">Detected Words</p>
            <div className="pills-row">
              {words.length === 0
                ? <span className="pill-empty">Sign something to map words here…</span>
                : words.map((w, i) => <span key={i} className="pill pill-word">{w}</span>)
              }
            </div>
          </div>

          {/* ── Sentence Output Panel ── */}
          {(sentence || loading) && (
            <div className="panel fade-in">
              <p className="panel-title">Formed Sentence</p>
              <div className={`sentence-box ${sentence ? '' : 'empty'}`}>
                {loading ? 'Forming a natural sentence and checking grammar…' : sentence}
              </div>

              {audio && (
                <div className="audio-wrap" style={{ marginTop: '1rem' }}>
                  <audio
                    controls
                    src={audio}
                    {...(prefs.audioAutoPlay ? { autoPlay: true } : {})}
                  />
                  {!prefs.audioAutoPlay && (
                    <p className="audio-hint">Press play on the player to hear the sentence.</p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ── Translations Panel ── */}
          {Object.keys(translations).length > 0 && (
            <div className="panel fade-in">
              <p className="panel-title">Translations</p>
              <div className="lang-tabs">
                {LANGUAGES.map(lang => (
                  <button
                    key={lang}
                    className={`lang-tab ${activeLang === lang ? 'active' : ''}`}
                    onClick={() => setActiveLang(lang)}
                  >
                    {lang.charAt(0).toUpperCase() + lang.slice(1)}
                  </button>
                ))}
              </div>
              <div className="translation-text">{displayText || '—'}</div>
            </div>
          )}

        </div>
      </main>
    </>
  )
}
