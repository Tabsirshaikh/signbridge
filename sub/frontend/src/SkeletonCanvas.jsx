import { useEffect, useRef } from 'react'

const HAND_CONNECTIONS = [
  { group: 'thumb',  pts: [[0,1],[1,2],[2,3],[3,4]],       color: '#8B7355' },
  { group: 'index',  pts: [[0,5],[5,6],[6,7],[7,8]],       color: '#5C6B7A' },
  { group: 'middle', pts: [[0,9],[9,10],[10,11],[11,12]],  color: '#6B7D6A' },
  { group: 'ring',   pts: [[0,13],[13,14],[14,15],[15,16]],color: '#7A8B8B' },
  { group: 'pinky',  pts: [[0,17],[17,18],[18,19],[19,20]],color: '#8B6B6B' },
  { group: 'palm',   pts: [[5,9],[9,13],[13,17]],          color: '#A39B8E' },
]

const BODY_CONNECTIONS = [
  [11,12],[11,23],[12,24],[23,24],
  [11,13],[13,15],
  [12,14],[14,16],
]

const CANVAS_SIZE = 400
const PADDING     = 50

function drawFrame(ctx, frameData) {
  ctx.fillStyle = '#F4F0E8'
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)

  if (!frameData || !frameData.pose) return

  const { pose, lh, rh } = frameData

  const allPts = []
  for (let i=0; i<=24; i++) {
    if (pose[i] && (pose[i].x !== 0 || pose[i].y !== 0)) allPts.push(pose[i])
  }
  lh.forEach(p => { if (p.x !== 0 || p.y !== 0) allPts.push(p) })
  rh.forEach(p => { if (p.x !== 0 || p.y !== 0) allPts.push(p) })

  if (allPts.length < 5) return

  const xs = allPts.map(p => p.x)
  const ys = allPts.map(p => p.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)

  const rangeX = Math.max(maxX - minX, 0.1)
  const rangeY = Math.max(maxY - minY, 0.1)
  const range = Math.max(rangeX, rangeY)

  const drawW = CANVAS_SIZE - PADDING * 2
  const offX = (range - rangeX) / 2
  const offY = (range - rangeY) / 2

  const toCanvas = (p) => ({
    x: PADDING + ((p.x - minX + offX) / range) * drawW,
    y: PADDING + ((p.y - minY + offY) / range) * drawW,
  })

  const cPose = pose.map(p => (p && (p.x !== 0 || p.y !== 0)) ? toCanvas(p) : null)
  const cLh   = lh.map(p => (p && (p.x !== 0 || p.y !== 0)) ? toCanvas(p) : null)
  const cRh   = rh.map(p => (p && (p.x !== 0 || p.y !== 0)) ? toCanvas(p) : null)

  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  ctx.strokeStyle = '#7A5C52'
  ctx.lineWidth = 3.5
  ctx.beginPath()
  const nose = cPose[0], lEar = cPose[7], rEar = cPose[8]
  if (nose && lEar && rEar) {
    const dx = lEar.x - rEar.x
    const ext = Math.abs(dx) * 1.5
    ctx.ellipse(nose.x, nose.y, ext * 0.45, ext * 0.6, 0, 0, Math.PI * 2)
    ctx.stroke()
    const le = cPose[2], re = cPose[5]
    if (le && re) {
      ctx.beginPath()
      ctx.moveTo(le.x - 5, le.y)
      ctx.lineTo(le.x + 5, le.y)
      ctx.moveTo(re.x - 5, re.y)
      ctx.lineTo(re.x + 5, re.y)
      ctx.stroke()
    }
    const lMouth = cPose[9], rMouth = cPose[10]
    if (lMouth && rMouth) {
      ctx.beginPath()
      ctx.moveTo(rMouth.x, rMouth.y)
      ctx.lineTo(lMouth.x, lMouth.y)
      ctx.stroke()
    }
  }

  ctx.strokeStyle = '#6B5D4F'
  ctx.lineWidth = 4.5
  ctx.beginPath()
  for (const [a, b] of BODY_CONNECTIONS) {
    if (cPose[a] && cPose[b]) {
      ctx.moveTo(cPose[a].x, cPose[a].y)
      ctx.lineTo(cPose[b].x, cPose[b].y)
    }
  }
  ctx.stroke()

  const drawHand = (handPts) => {
    ctx.lineWidth = 3
    for (const link of HAND_CONNECTIONS) {
      ctx.strokeStyle = link.color
      ctx.beginPath()
      for (const [a, b] of link.pts) {
        if (handPts[a] && handPts[b]) {
          ctx.moveTo(handPts[a].x, handPts[a].y)
          ctx.lineTo(handPts[b].x, handPts[b].y)
        }
      }
      ctx.stroke()
    }
  }

  drawHand(cLh)
  drawHand(cRh)
}

export default function SkeletonCanvas({ frames = [], fps = 10, onComplete, paused = false }) {
  const canvasRef  = useRef(null)
  const rafRef     = useRef(null)
  const frameIndex = useRef(0)
  const lastTime   = useRef(0)

  useEffect(() => {
    frameIndex.current = 0
    lastTime.current = 0
    const canvas = canvasRef.current
    if (canvas && frames?.length) {
      const ctx = canvas.getContext('2d')
      drawFrame(ctx, frames[0])
    }
  }, [frames])

  useEffect(() => {
    if (!frames?.length) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    if (paused) {
      const idx = Math.min(Math.max(0, frameIndex.current), frames.length - 1)
      drawFrame(ctx, frames[idx])
      return undefined
    }

    lastTime.current = 0
    const interval = 1000 / Math.max(4, Math.min(fps, 24))

    const animate = (timestamp) => {
      if (!lastTime.current) lastTime.current = timestamp
      const dt = timestamp - lastTime.current

      if (dt >= interval) {
        lastTime.current = timestamp - (dt % interval)
        const idx = frameIndex.current
        if (idx < frames.length) {
          drawFrame(ctx, frames[idx])
          frameIndex.current++
        } else {
          cancelAnimationFrame(rafRef.current)
          if (onComplete) onComplete()
          return
        }
      }
      rafRef.current = requestAnimationFrame(animate)
    }

    rafRef.current = requestAnimationFrame(animate)

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [frames, fps, onComplete, paused])

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_SIZE}
      height={CANVAS_SIZE}
      style={{ display: 'block', borderRadius: '14px', background: '#F4F0E8', border: '1px solid rgba(74, 66, 58, 0.12)' }}
    />
  )
}
