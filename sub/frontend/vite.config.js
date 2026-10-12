import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      // REST API calls
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      // Socket.IO WebSocket + polling handshake
      '/socket.io': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        ws: true,          // proxy WebSocket upgrades
      }
    }
  }
})
