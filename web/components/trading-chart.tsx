"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface TradingChartProps {
  symbol: string
}

export function TradingChart({ symbol }: TradingChartProps) {
  const [timeframe, setTimeframe] = useState("1h")
  const [chartData, setChartData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchChartData = async () => {
      setLoading(true)
      try {
        // Simulate chart data - replace with actual API call
        const data = Array.from({ length: 100 }, (_, i) => ({
          time: new Date(Date.now() - (100 - i) * 60000).toISOString(),
          price: 50000 + Math.random() * 10000,
          volume: Math.random() * 1000000,
        }))
        setChartData(data)
      } catch (error) {
        console.error("Error fetching chart data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchChartData()
  }, [symbol, timeframe])

  const timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

  return (
    <Card className="chart-container h-96">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{symbol}</CardTitle>
          <div className="flex gap-1">
            {timeframes.map((tf) => (
              <Button
                key={tf}
                variant={timeframe === tf ? "default" : "ghost"}
                size="sm"
                onClick={() => setTimeframe(tf)}
                className="text-xs px-2 py-1 h-7"
              >
                {tf}
              </Button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant="outline" className="text-xs">
            ${chartData[chartData.length - 1]?.price?.toFixed(2) || "0.00"}
          </Badge>
          <Badge variant="outline" className="text-xs text-success">
            +2.34%
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0 h-80">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" />
              <XAxis
                dataKey="time"
                stroke="rgb(var(--muted-foreground))"
                fontSize={12}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis stroke="rgb(var(--muted-foreground))" fontSize={12} domain={["dataMin", "dataMax"]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgb(var(--card))",
                  border: "1px solid rgb(var(--border))",
                  borderRadius: "6px",
                }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value) => [`$${value.toFixed(2)}`, "Price"]}
              />
              <Line type="monotone" dataKey="price" stroke="rgb(var(--primary))" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}
