import { useEffect } from 'react'
import { useComfort } from './ComfortContext'

export default function ComfortPanel() {
  const { prefs, setPref, panelOpen, closeComfortPanel, openComfortPanel } = useComfort()

  useEffect(() => {
    if (!panelOpen) return
    const onKey = (e) => {
      if (e.key === 'Escape') closeComfortPanel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [panelOpen, closeComfortPanel])

  if (!panelOpen) {
    return (
      <button
        type="button"
        className="comfort-fab"
        onClick={openComfortPanel}
        aria-expanded="false"
        aria-controls="comfort-panel-dialog"
        title="Comfort and accessibility"
      >
        Comfort
      </button>
    )
  }

  return (
    <>
      <button
        type="button"
        className="comfort-fab comfort-fab--active"
        onClick={closeComfortPanel}
        aria-expanded="true"
        aria-controls="comfort-panel-dialog"
      >
        Close
      </button>
      <div
        className="comfort-backdrop"
        aria-hidden="true"
        onClick={closeComfortPanel}
      />
      <div
        id="comfort-panel-dialog"
        className="comfort-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="comfort-dialog-title"
      >
        <h2 id="comfort-dialog-title" className="comfort-dialog-title">
          Comfort &amp; access
        </h2>
        <p className="comfort-dialog-intro">
          Adjust these anytime. Your choices are saved on this device only.
        </p>

        <ul className="comfort-options">
          <li className="comfort-row">
            <label className="comfort-label">
              <input
                type="checkbox"
                checked={prefs.largeText}
                onChange={(e) => setPref('largeText', e.target.checked)}
              />
              <span>Larger text across the app</span>
            </label>
          </li>
          <li className="comfort-row">
            <label className="comfort-label">
              <input
                type="checkbox"
                checked={prefs.reducedMotion}
                onChange={(e) => setPref('reducedMotion', e.target.checked)}
              />
              <span>Reduce motion (softer animations, slower signs)</span>
            </label>
          </li>
          <li className="comfort-row">
            <label className="comfort-label">
              <input
                type="checkbox"
                checked={prefs.highContrast}
                onChange={(e) => setPref('highContrast', e.target.checked)}
              />
              <span>Stronger contrast for text and borders</span>
            </label>
          </li>
          <li className="comfort-row">
            <label className="comfort-label">
              <input
                type="checkbox"
                checked={prefs.largerSignCanvas}
                onChange={(e) => setPref('largerSignCanvas', e.target.checked)}
              />
              <span>Larger sign animation (gesture view)</span>
            </label>
          </li>
          <li className="comfort-row">
            <label className="comfort-label">
              <input
                type="checkbox"
                checked={prefs.audioAutoPlay}
                onChange={(e) => setPref('audioAutoPlay', e.target.checked)}
              />
              <span>Play spoken sentence automatically (gesture view)</span>
            </label>
          </li>
        </ul>

        <div className="comfort-field">
          <span className="comfort-field-label" id="sign-speed-label">
            Sign animation pace
          </span>
          <select
            className="comfort-select"
            aria-labelledby="sign-speed-label"
            value={prefs.signSpeed}
            onChange={(e) => setPref('signSpeed', e.target.value)}
          >
            <option value="gentle">Gentle (slowest)</option>
            <option value="standard">Standard</option>
            <option value="deliberate">Very deliberate</option>
          </select>
        </div>

        <button
          type="button"
          className="comfort-tips-toggle"
          aria-expanded={prefs.showTips}
          onClick={() => setPref('showTips', !prefs.showTips)}
        >
          {prefs.showTips ? 'Hide tips' : 'Show comfort tips'}
        </button>
        {prefs.showTips && (
          <ul className="comfort-tips">
            <li>Use even lighting on your face and hands for clearer camera reading.</li>
            <li>Keep your upper body in frame; step back slightly if signs are clipped.</li>
            <li>Pause the sign animation if you need more time to study a gesture.</li>
            <li>Keyboard: Enter completes a sentence, Backspace removes the last word, Q returns home.</li>
            <li>Rest your eyes and shoulders between longer sessions.</li>
          </ul>
        )}

        <button type="button" className="btn btn-primary comfort-done" onClick={closeComfortPanel}>
          Done
        </button>
      </div>
    </>
  )
}
