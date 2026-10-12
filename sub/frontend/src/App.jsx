import { useState, useEffect } from 'react'
import { AnimatePresence } from 'framer-motion'
import Direction1 from './Direction1'
import Direction2 from './Direction2'
import ComfortPanel from './ComfortPanel.jsx'
import SplashScreen from './SplashScreen.jsx'
import HomeScreen from './HomeScreen.jsx'

export default function App() {
  const [screen, setScreen] = useState('splash')

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape' && screen === 'splash') {
        setScreen('home')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [screen])

  return (
    <>
      <AnimatePresence mode="wait">
        {screen === 'splash' && (
          <SplashScreen key="splash" onEnter={() => setScreen('home')} />
        )}
        {screen === 'home' && (
          <HomeScreen
            key="home"
            onPickD1={() => setScreen('d1')}
            onPickD2={() => setScreen('d2')}
          />
        )}
      </AnimatePresence>

      {screen === 'd1' && <Direction1 onBack={() => setScreen('home')} />}
      {screen === 'd2' && <Direction2 onBack={() => setScreen('home')} />}

      <ComfortPanel />
    </>
  )
}
