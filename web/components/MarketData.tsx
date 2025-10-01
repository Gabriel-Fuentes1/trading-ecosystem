"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface MarketDataProps {
  symbol: string
}

export function MarketData({ symbol }: MarketDataProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Market Data - {symbol}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-muted-foreground">Last Price</div>
            <div className="text-2xl font-bold">$0.00</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">24h Change</div>
            <div className="text-2xl font-bold text-green-600">+0.00%</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">24h High</div>
            <div className="text-lg font-semibold">$0.00</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">24h Low</div>
            <div className="text-lg font-semibold">$0.00</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">24h Volume</div>
            <div className="text-lg font-semibold">0</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
