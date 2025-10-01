"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface OrderBookProps {
  symbol: string
}

export function OrderBook({ symbol }: OrderBookProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Order Book - {symbol}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="grid grid-cols-3 gap-2 text-sm font-medium text-muted-foreground">
            <div>Price</div>
            <div className="text-right">Size</div>
            <div className="text-right">Total</div>
          </div>
          <div className="space-y-1">
            {/* Placeholder order book data */}
            <div className="grid grid-cols-3 gap-2 text-sm text-green-600">
              <div>50,000</div>
              <div className="text-right">0.5</div>
              <div className="text-right">25,000</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
