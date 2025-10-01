"use client"

import { useState, useEffect } from "react"

export function useDashboardData() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Simulate API call
        await new Promise((resolve) => setTimeout(resolve, 1000))
        setData({
          portfolio_value: 1250000,
          daily_pnl: 15420,
          daily_pnl_percentage: 1.25,
          total_positions: 12,
          active_orders: 5,
          top_performers: [],
          recent_trades: [],
          risk_metrics: {
            var: 25000,
            sharpe_ratio: 2.1,
            max_drawdown: 8.5,
          },
        })
      } catch (err) {
        setError(err as Error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  return { data, loading, error }
}
