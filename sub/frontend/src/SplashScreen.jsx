import { useEffect } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { useComfort } from './ComfortContext.jsx'

const letterEase = [0.22, 1, 0.36, 1]

export default function SplashScreen({ onEnter }) {
  const { prefs } = useComfort()
  const reduce = useReducedMotion() || prefs.reducedMotion

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Enter' || e.repeat) return
      const el = document.activeElement
      const tag = el?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el?.closest?.('[role="dialog"]')) return
      onEnter()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onEnter])
  const title = 'SignBridge'
  const dur = reduce ? 0.01 : 0.65
  const stagger = reduce ? 0 : 0.045

  return (
    <motion.main
      className="splash"
      id="splash-main"
      role="main"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, filter: 'blur(8px)' }}
      transition={{ duration: reduce ? 0.15 : 0.45 }}
    >
      <a href="#splash-enter" className="skip-link">
        Skip to enter
      </a>

      <div className="home__bg splash__bg" aria-hidden="true">
        <span className="home__blob home__blob--1" />
        <span className="home__blob home__blob--2" />
        <span className="home__blob home__blob--3" />
      </div>

      <div className="splash__inner">
        <motion.p
          className="splash__eyebrow"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: dur, delay: reduce ? 0 : 0.1, ease: letterEase }}
        >
          Sign language · two directions
        </motion.p>

        <h1 className="splash__title" aria-label={title}>
          {title.split('').map((ch, i) => (
            <motion.span
              key={`${ch}-${i}`}
              className="splash__char"
              initial={{ opacity: 0, y: reduce ? 0 : 48, rotateX: reduce ? 0 : -25 }}
              animate={{ opacity: 1, y: 0, rotateX: 0 }}
              transition={{
                duration: dur,
                delay: reduce ? 0 : 0.15 + i * stagger,
                ease: letterEase,
              }}
            >
              {ch}
            </motion.span>
          ))}
        </h1>

        <motion.p
          className="splash__lede"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: dur, delay: reduce ? 0 : 0.65, ease: letterEase }}
        >
          Understand and be understood—gesture to text, or text to gesture—with calm, on-device vision first.
        </motion.p>

        <motion.div
          className="splash__cta-wrap"
          initial={{ opacity: 0, scale: reduce ? 1 : 0.92 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: dur, delay: reduce ? 0 : 0.95, ease: letterEase }}
        >
          <motion.button
            type="button"
            id="splash-enter"
            className="splash__enter"
            onClick={onEnter}
            whileHover={reduce ? {} : { scale: 1.03 }}
            whileTap={reduce ? {} : { scale: 0.98 }}
          >
            <span className="splash__enter-glow" aria-hidden="true" />
            <span className="splash__enter-label">Enter</span>
            <span className="splash__enter-arrow" aria-hidden="true">
              →
            </span>
          </motion.button>
          <p className="splash__hint">Press Enter when focused, or use the skip link</p>
        </motion.div>
      </div>
    </motion.main>
  )
}
