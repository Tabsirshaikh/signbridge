/**
 * SignDisplay.jsx — keyword queue + skeleton playback
 */
import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import SkeletonCanvas from './SkeletonCanvas'
import { useComfort } from './ComfortContext'

function wordHasAnimation(word, keypoints) {
  const upper = word.toUpperCase()
  if (keypoints[upper]) return true
  for (const ch of upper) {
    if (keypoints[ch]) return true
  }
  return false
}

function buildSignQueue(keywords, keypoints) {
  const queue = []
  keywords.forEach((word, wordIdx) => {
    const upper = word.toUpperCase()
    if (keypoints[upper]) {
      queue.push({ sign: upper, frames: keypoints[upper], wordIdx, isSpell: false })
    } else {
      for (const ch of upper) {
        if (keypoints[ch]) {
          queue.push({ sign: ch, frames: keypoints[ch], wordIdx, isSpell: true })
        }
      }
    }
  })
  return queue
}

export default function SignDisplay({ keywords = [], keypoints = {} }) {
  const { signTiming } = useComfort()
  const { fps: PLAYBACK_FPS, pauseMs: PAUSE_MS } = signTiming

  const [queue, setQueue]       = useState([])
  const [queueIdx, setQueueIdx] = useState(0)
  const [pausing, setPausing]   = useState(false)
  const [missingKw, setMissingKw] = useState([])
  const [userHold, setUserHold] = useState(false)

  const timerRef = useRef(null)

  useEffect(() => {
    setUserHold(false)
  }, [queueIdx])

  useEffect(() => {
    if (keywords.length === 0 || Object.keys(keypoints).length === 0) {
      setQueue([])
      setMissingKw([])
      return
    }
    const missing = keywords.filter(w => !wordHasAnimation(w, keypoints))
    setMissingKw(missing)
    const q = buildSignQueue(keywords, keypoints)
    setQueue(q)
    setQueueIdx(0)
    setPausing(false)
    setUserHold(false)
  }, [keywords, keypoints])

  const handleSignComplete = useCallback(() => {
    setPausing(true)
    timerRef.current = setTimeout(() => {
      setPausing(false)
      setQueueIdx(prev => prev + 1)
    }, PAUSE_MS)
  }, [PAUSE_MS])

  const handleReplay = useCallback(() => {
    setQueueIdx(0)
    setPausing(false)
    setUserHold(false)
  }, [])

  useEffect(() => () => clearTimeout(timerRef.current), [])

  const labelText = useMemo(() => {
    if (queue.length === 0) return ''
    const current = queue[queueIdx]
    const finished = queueIdx >= queue.length
    if (finished) return 'Sequence complete'
    if (pausing) return 'Brief pause before next sign'
    return current?.sign ? `Now showing: ${current.sign}` : ''
  }, [queue, queueIdx, pausing])

  if (queue.length === 0 && missingKw.length > 0) {
    return (
      <div className="sign-section fade-in">
        <p style={{ fontSize: '0.95rem', color: 'var(--text-dim)', lineHeight: 1.5 }}>
          No skeleton animation is available for these keywords (they are not in the 250-sign keypoint set):{' '}
          <strong>{missingKw.join(', ')}</strong>
        </p>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-dim)', marginTop: '0.75rem' }}>
          Try rephrasing using simpler words that appear in the model vocabulary, or type words you know are signed in the dataset.
        </p>
      </div>
    )
  }

  if (queue.length === 0) return null

  const current     = queue[queueIdx]
  const finished    = queueIdx >= queue.length
  const progress    = finished ? 100 : Math.round((queueIdx / queue.length) * 100)

  const wordStatus = keywords.map((_, wi) => {
    const signs = queue.filter(s => s.wordIdx === wi)
    if (signs.length === 0) return 'skip'
    const firstIdx = queue.findIndex(s => s.wordIdx === wi)
    const lastIdx  = firstIdx + signs.length - 1
    if (queueIdx > lastIdx)         return 'done'
    if (queueIdx >= firstIdx)        return queue[firstIdx].isSpell ? 'spelling' : 'active'
    return 'waiting'
  })

  return (
    <div className="sign-section fade-in">
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {labelText}
      </div>

      <div className="pills-row" style={{ justifyContent: 'center' }}>
        {keywords.map((word, i) => {
          const status = wordStatus[i]
          const cls =
            status === 'active'   ? 'pill pill-active' :
            status === 'spelling' ? 'pill pill-spell'  :
            status === 'done'     ? 'pill pill-done'   :
            'pill pill-wait'
          return (
            <span key={i} id={`word-pill-${i}`} className={cls}>
              {status === 'spelling' ? 'Spell: ' : ''}{word}
            </span>
          )
        })}
      </div>

      <div className="sign-word-label" aria-hidden="true">
        {finished ? 'Done' : (pausing ? 'Pause' : (current?.sign || ''))}
      </div>

      {!finished && (
        <>
          <div className={`canvas-wrap ${userHold ? 'canvas-wrap--paused' : ''}`}>
            <SkeletonCanvas
              key={`${queueIdx}`}
              frames={current?.frames || []}
              fps={PLAYBACK_FPS}
              paused={userHold}
              onComplete={handleSignComplete}
            />
          </div>
          <div className="sign-playback-controls">
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setUserHold(true)}
              disabled={finished || userHold}
            >
              Pause movement
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setUserHold(false)}
              disabled={!userHold}
            >
              Resume
            </button>
          </div>
        </>
      )}

      {finished && (
        <button type="button" className="btn btn-primary" onClick={handleReplay} style={{ marginTop: '1rem' }}>
          Replay sequence
        </button>
      )}

      <div className="progress-bar-wrap">
        <div className="progress-bar" style={{ width: `${progress}%` }} />
      </div>
      <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
        {finished ? 'Animation complete' : `Sign ${queueIdx + 1} of ${queue.length}`}
      </span>
    </div>
  )
}
