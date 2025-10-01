"use client"

import { useState, useEffect } from "react"

export function useWebSocket() {
  const [marketData, setMarketData] = useState<any>({})
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    // Simulate WebSocket connection
    setIsConnected(true)

    // Simulate market data updates
    const interval = setInterval(() => {
      setMarketData({
        BTCUSDT: { price: 50000 + Math.random() * 1000, change: Math.random() * 5 - 2.5 },
        ETHUSDT: { price: 3000 + Math.random() * 100, change: Math.random() * 5 - 2.5 },
        BNBUSDT: { price: 400 + Math.random() * 20, change: Math.random() * 5 - 2.5 },
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  return { marketData, isConnected }
}
