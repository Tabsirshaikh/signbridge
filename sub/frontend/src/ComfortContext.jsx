import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react'

const STORAGE_KEY = 'signbridge-comfort-v1'

const DEFAULT = {
  largeText: false,
  reducedMotion: false,
  highContrast: false,
  signSpeed: 'standard',
  audioAutoPlay: true,
  largerSignCanvas: false,
  showTips: false,
}

const SPEED_MAP = {
  gentle: { fps: 6, pauseMs: 950 },
  standard: { fps: 9, pauseMs: 700 },
  deliberate: { fps: 5, pauseMs: 1200 },
}

const ComfortContext = createContext(null)

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const p = JSON.parse(raw)
    return { ...DEFAULT, ...p }
  } catch {
    return null
  }
}

export function ComfortProvider({ children }) {
  const [prefs, setPrefs] = useState(() => {
    const stored = loadStored()
    if (stored) return stored
    const next = { ...DEFAULT }
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      next.reducedMotion = true
    }
    return next
  })

  const [panelOpen, setPanelOpen] = useState(false)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs))
    } catch {
      /* ignore */
    }
  }, [prefs])

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('comfort-large-text', prefs.largeText)
    root.classList.toggle('comfort-reduced-motion', prefs.reducedMotion)
    root.classList.toggle('comfort-high-contrast', prefs.highContrast)
    root.classList.toggle('comfort-large-sign', prefs.largerSignCanvas)
  }, [prefs.largeText, prefs.reducedMotion, prefs.highContrast, prefs.largerSignCanvas])

  const setPref = useCallback((key, value) => {
    setPrefs((prev) => ({ ...prev, [key]: value }))
  }, [])

  const signTiming = useMemo(() => {
    const base = SPEED_MAP[prefs.signSpeed] || SPEED_MAP.standard
    if (prefs.reducedMotion) {
      return { fps: Math.min(base.fps, 5), pauseMs: Math.max(base.pauseMs, 1000) }
    }
    return base
  }, [prefs.signSpeed, prefs.reducedMotion])

  const value = useMemo(
    () => ({
      prefs,
      setPref,
      signTiming,
      panelOpen,
      setPanelOpen,
      openComfortPanel: () => setPanelOpen(true),
      closeComfortPanel: () => setPanelOpen(false),
    }),
    [prefs, setPref, signTiming, panelOpen]
  )

  return <ComfortContext.Provider value={value}>{children}</ComfortContext.Provider>
}

export function useComfort() {
  const ctx = useContext(ComfortContext)
  if (!ctx) throw new Error('useComfort must be used within ComfortProvider')
  return ctx
}
