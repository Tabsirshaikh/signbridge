/**
 * Direction2.jsx  ─  Text to Sign  (Hearing → Deaf)
 * ────────────────────────────────────────────────────
 * 1. Language selector + textarea or hold-to-speak mic
 * 2. Send → POST /text-to-sign → English text + keywords
 * 3. Show English message in large font
 * 4. "Show in Sign Language" → SignDisplay animates keywords
 */
import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import axios from 'axios'
import SignDisplay from './SignDisplay'

const API = 'http://localhost:5000'

/** Base64 without spreading huge buffers (avoids stack errors on long recordings). */
function uint8ToBase64(u8) {
  const CHUNK = 0x8000
  let binary = ''
  for (let i = 0; i < u8.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, u8.subarray(i, i + CHUNK))
  }
  return btoa(binary)
}

// Language config: { label, code, bcp47 }
const LANG_OPTIONS = [
  { label: 'English',  code: 'en',    bcp47: 'en-US'  },
  { label: 'Hindi',    code: 'hi',    bcp47: 'hi-IN'  },
  { label: 'Marathi',  code: 'mr',    bcp47: 'mr-IN'  },
  { label: 'Spanish',  code: 'es',    bcp47: 'es-ES'  },
  { label: 'French',   code: 'fr',    bcp47: 'fr-FR'  },
  { label: 'Arabic',   code: 'ar',    bcp47: 'ar-SA'  },
]

export default function Direction2({ onBack }) {
  const [selectedLang,  setSelectedLang]  = useState(LANG_OPTIONS[0])
  const [inputText,     setInputText]     = useState('')
  const [englishText,   setEnglishText]   = useState('')
  const [keywords,      setKeywords]      = useState([])
  const [showSigns,     setShowSigns]     = useState(false)
  const [loading,       setLoading]       = useState(false)
  const [recording,     setRecording]     = useState(false)
  const [keypoints,     setKeypoints]     = useState({})
  const [kpLoaded,      setKpLoaded]      = useState(false)
  const [kpError,       setKpError]       = useState(false)

  const mediaRecRef     = useRef(null)
  const chunksRef       = useRef([])
  const browserRecRef   = useRef(null)
  const transcriptRef   = useRef('')

  const browserSttAvailable =
    typeof window !== 'undefined' &&
    !!(window.SpeechRecognition || window.webkitSpeechRecognition)

  // ── Load asl_keypoints.json lazily ──────────────────────────────────────────
  useEffect(() => {
    import('./data/asl_keypoints.json')
      .then(mod => {
        setKeypoints(mod.default || mod)
        setKpLoaded(true)
      })
      .catch(() => {
        setKpError(true)
        console.warn('asl_keypoints.json not found. Run extract_keypoints.py first.')
      })
  }, [])

  // ── Send text ───────────────────────────────────────────────────────────────
  const handleSend = useCallback(async () => {
    const text = inputText.trim()
    if (!text) return

    setLoading(true)
    setEnglishText('')
    setKeywords([])
    setShowSigns(false)

    try {
      const res = await axios.post(`${API}/text-to-sign`, {
        text,
        lang: selectedLang.code,
      })
      setEnglishText(res.data.english_text || text)
      setKeywords(res.data.keywords || [])
    } catch (e) {
      // Fallback offline: just show the text as-is
      setEnglishText(text)
      setKeywords(text.toLowerCase().split(/\s+/).filter(Boolean))
    }
    setLoading(false)
  }, [inputText, selectedLang])

  // ── Mic: Hold to Speak ──────────────────────────────────────────────────────
  // Prefer Web Speech API (Chrome/Edge): no ffmpeg on the server. Fallback: MediaRecorder + backend.
  const startRecording = useCallback(async () => {
    const SR = typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition)
    if (SR) {
      if (browserRecRef.current) return
      transcriptRef.current = ''
      const rec = new SR()
      rec.lang = selectedLang.bcp47
      rec.continuous = true
      rec.interimResults = true
      rec.onresult = (event) => {
        let line = ''
        for (let i = event.resultIndex; i < event.results.length; i++) {
          line += event.results[i][0].transcript
        }
        if (line) transcriptRef.current = line.trim()
      }
      rec.onerror = (ev) => {
        console.warn('Speech recognition:', ev.error)
      }
      rec.onend = () => {
        browserRecRef.current = null
        const t = transcriptRef.current.trim()
        if (t) setInputText(t)
        setRecording(false)
      }
      try {
        rec.start()
        browserRecRef.current = rec
        setRecording(true)
      } catch (e) {
        console.warn(e)
        browserRecRef.current = null
        setRecording(false)
      }
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream)
      chunksRef.current = []
      mr.ondataavailable = e => chunksRef.current.push(e.data)
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const arrayBuf = await blob.arrayBuffer()
        const b64 = uint8ToBase64(new Uint8Array(arrayBuf))

        setLoading(true)
        try {
          const res = await axios.post(`${API}/speech-to-text`, {
            audio_b64: b64,
            lang: selectedLang.bcp47,
            audio_format: 'webm',
          })
          if (res.data.text) setInputText(res.data.text)
        } catch (e) {
          const err = e.response?.data?.error || e.message
          console.warn('Speech-to-text:', err)
          if (typeof err === 'string' && err.toLowerCase().includes('ffmpeg')) {
            alert('Server needs ffmpeg on PATH for this browser, or use Chrome/Edge for hold-to-speak (browser speech).')
          }
        }
        setLoading(false)
      }
      mr.start()
      mediaRecRef.current = mr
      setRecording(true)
    } catch (e) {
      alert('Microphone access denied.')
    }
  }, [selectedLang])

  const stopRecording = useCallback(() => {
    if (browserRecRef.current) {
      try {
        browserRecRef.current.stop()
      } catch (_) {
        browserRecRef.current = null
        setRecording(false)
      }
      return
    }
    mediaRecRef.current?.stop()
    mediaRecRef.current = null
    setRecording(false)
  }, [])

  // ── Keyboard send ───────────────────────────────────────────────────────────
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  const liveAnnouncement = useMemo(() => {
    if (loading) return 'Processing your message.'
    if (englishText) return `English line: ${englishText}.`
    if (keywords.length) return `${keywords.length} sign keywords ready. Open comfort panel to slow animation or pause.`
    return 'Language to gesture. Pick a language, type or speak, then send.'
  }, [loading, englishText, keywords.length])

  return (
    <>
      <a href="#script-main" className="skip-link">
        Skip to main content
      </a>
      <main id="script-main" className="screen" tabIndex={-1}>
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {liveAnnouncement}
      </div>
      <header className="screen-header">
        <button id="d2-back" className="back-btn" onClick={onBack} title="Back">←</button>
        <h1 className="screen-title">Language to gesture</h1>
        <span className="screen-tag">Type, speak, then watch the signs</span>
      </header>

      <div className="screen-body">
        {/* ── Language Selector ── */}
        <div className="panel">
          <p className="panel-title">Input Language</p>
          <div className="lang-tabs">
            {LANG_OPTIONS.map(lang => (
              <button
                key={lang.code}
                id={`lang-${lang.code}`}
                className={`lang-tab ${selectedLang.code === lang.code ? 'active' : ''}`}
                onClick={() => setSelectedLang(lang)}
              >
                {lang.label}
              </button>
            ))}
          </div>
        </div>

        {/* ── Input ── */}
        <div className="panel">
          <p className="panel-title">Your Message</p>
          <textarea
            id="text-input"
            className="textarea-input"
            placeholder={`Type your message in ${selectedLang.label}…\n(Press Enter to send, Shift+Enter for newline)`}
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={4}
          />

          <div className="btn-row" style={{ marginTop: '1rem' }}>
            <button
              id="btn-send"
              className="btn btn-primary btn-lg"
              onClick={handleSend}
              disabled={!inputText.trim() || loading}
            >
              {loading ? <><span className="spinner" /> Processing…</> : '📤 Send'}
            </button>

            <button
              id="btn-mic"
              className={`btn btn-ghost btn-lg mic-btn ${recording ? 'recording' : ''}`}
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onMouseLeave={recording ? stopRecording : undefined}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              title={
                browserSttAvailable
                  ? 'Hold to speak (browser speech, no server ffmpeg)'
                  : 'Hold to speak (server needs ffmpeg for this browser)'
              }
            >
              {recording ? '🔴 Recording…' : '🎤 Hold to Speak'}
            </button>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '0.5rem' }}>
            {browserSttAvailable
              ? 'Speech is recognized in your browser (Chrome/Edge). No ffmpeg required.'
              : 'This browser uses server speech-to-text; install ffmpeg on the server or try Chrome/Edge.'}
          </p>
        </div>

        {/* ── Message Display ── */}
        {(englishText || loading) && (
          <div className="panel fade-in">
            <p className="panel-title">Message for Deaf Person (English)</p>
            <div className="message-display">
              {loading ? 'Translating…' : englishText}
            </div>

            {!loading && keywords.length > 0 && (
              <>
                {/* Keyword pills */}
                <div className="pills-row" style={{ marginTop: '1rem' }}>
                  {keywords.map((kw, i) => (
                    <span key={i} className="pill pill-word">
                      {kw}
                    </span>
                  ))}
                </div>

                {/* Show Sign button */}
                {!showSigns && (
                  <div style={{ marginTop: '1.25rem' }}>
                    {kpError ? (
                      <p style={{ fontSize: '0.85rem', color: 'var(--orange)' }}>
                        ⚠️ asl_keypoints.json not found. Run <code>python extract_keypoints.py</code> first.
                      </p>
                    ) : (
                      <button
                        id="btn-show-signs"
                        className="btn btn-green"
                        onClick={() => setShowSigns(true)}
                        disabled={!kpLoaded}
                      >
                        {kpLoaded ? 'View sign animation' : 'Loading keypoints…'}
                      </button>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ── Sign Animation ── */}
        {showSigns && keywords.length > 0 && kpLoaded && (
          <div className="panel fade-in">
            <p className="panel-title">Sign Language Animation</p>
            <SignDisplay keywords={keywords} keypoints={keypoints} />
          </div>
        )}
      </div>
    </main>
    </>
  )
}
