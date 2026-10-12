import { motion, useReducedMotion } from 'framer-motion'
import { useComfort } from './ComfortContext.jsx'

function IconCamera() {
  return (
    <svg className="home__ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden="true">
      <path d="M4 9a2 2 0 012-2h2.5L10 5h4l1.5 2H19a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2V9z" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="12" cy="13" r="3.25" />
    </svg>
  )
}

function IconPen() {
  return (
    <svg className="home__ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden="true">
      <path d="M12 20h9M16.5 3.5a2.12 2.12 0 013 3L8 18l-4 1 1-4L16.5 3.5z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

const ease = [0.22, 1, 0.36, 1]

export default function HomeScreen({ onPickD1, onPickD2 }) {
  const { prefs } = useComfort()
  const reduce = useReducedMotion() || prefs.reducedMotion
  const d = reduce ? 0.01 : 0.5
  const base = reduce ? 0 : 0.08

  return (
    <motion.main
      className="home"
      id="landing-main"
      role="main"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, y: reduce ? 0 : -16, filter: 'blur(6px)' }}
      transition={{ duration: reduce ? 0.12 : 0.4 }}
    >
      <a href="#path-choices" className="skip-link">
        Skip to path choices
      </a>

      <div className="home__bg" aria-hidden="true">
        <span className="home__blob home__blob--1" />
        <span className="home__blob home__blob--2" />
        <span className="home__blob home__blob--3" />
      </div>

      <div className="home__layout">
        <motion.header
          className="home__top"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: d, delay: base, ease }}
        >
          <span className="home__logo">SignBridge</span>
          <span className="home__tag">Local-first · private</span>
        </motion.header>

        <motion.section
          className="home__intro"
          aria-labelledby="home-title"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: d, delay: base + 0.06, ease }}
        >
          <p className="home__eyebrow">Choose your flow</p>
          <h1 id="home-title" className="home__title">
            How do you want to communicate?
          </h1>
          <p className="home__lead">
            One workspace, two directions: movement into language, or language into signs you can follow.
          </p>
        </motion.section>

        <div id="path-choices" className="home__grid" tabIndex={-1}>
          <motion.button
            type="button"
            id="card-sign-to-text"
            className="home__tile home__tile--a"
            onClick={onPickD1}
            initial={{ opacity: 0, x: reduce ? 0 : -36 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: d, delay: base + 0.18, ease }}
            whileHover={reduce ? {} : { y: -4, transition: { duration: 0.2 } }}
            whileTap={reduce ? {} : { scale: 0.99 }}
          >
            <span className="home__tile-shine" aria-hidden="true" />
            <span className="home__tile-icon" aria-hidden="true">
              <IconCamera />
            </span>
            <span className="home__tile-body">
              <span className="home__tile-kicker">I sign · camera reads me</span>
              <span className="home__tile-title">Gesture to language</span>
              <span className="home__tile-desc">
                Hands and posture through the lens—text, speech, and translations when you need them.
              </span>
            </span>
            <span className="home__tile-go">Open</span>
          </motion.button>

          <motion.button
            type="button"
            id="card-text-to-sign"
            className="home__tile home__tile--b"
            onClick={onPickD2}
            initial={{ opacity: 0, x: reduce ? 0 : 36 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: d, delay: base + 0.28, ease }}
            whileHover={reduce ? {} : { y: -4, transition: { duration: 0.2 } }}
            whileTap={reduce ? {} : { scale: 0.99 }}
          >
            <span className="home__tile-shine" aria-hidden="true" />
            <span className="home__tile-icon" aria-hidden="true">
              <IconPen />
            </span>
            <span className="home__tile-body">
              <span className="home__tile-kicker">I type or speak · I watch signs</span>
              <span className="home__tile-title">Language to gesture</span>
              <span className="home__tile-desc">
                Your line becomes sign vocabulary and a paced skeleton animation to mirror.
              </span>
            </span>
            <span className="home__tile-go">Open</span>
          </motion.button>
        </div>

        <motion.footer
          className="home__foot"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: d, delay: base + 0.45 }}
        >
          <p className="home__foot-note">
            MediaPipe on-device · optional cloud phrasing · Q returns to this hub
          </p>
        </motion.footer>
      </div>
    </motion.main>
  )
}
