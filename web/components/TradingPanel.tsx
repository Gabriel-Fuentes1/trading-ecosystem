"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface TradingPanelProps {
  symbol: string
}

export function TradingPanel({ symbol }: TradingPanelProps) {
  const [orderType, setOrderType] = useState<"market" | "limit">("market")
  const [side, setSide] = useState<"buy" | "sell">("buy")

  return (
    <Card>
      <CardHeader>
        <CardTitle>Place Order - {symbol}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex gap-2">
            <Button variant={side === "buy" ? "default" : "outline"} onClick={() => setSide("buy")} className="flex-1">
              Buy
            </Button>
            <Button
              variant={side === "sell" ? "default" : "outline"}
              onClick={() => setSide("sell")}
              className="flex-1"
            >
              Sell
            </Button>
          </div>

          <div className="flex gap-2">
            <Button
              variant={orderType === "market" ? "default" : "outline"}
              onClick={() => setOrderType("market")}
              className="flex-1"
            >
              Market
            </Button>
            <Button
              variant={orderType === "limit" ? "default" : "outline"}
              onClick={() => setOrderType("limit")}
              className="flex-1"
            >
              Limit
            </Button>
          </div>

          {orderType === "limit" && (
            <div className="space-y-2">
              <Label htmlFor="price">Price</Label>
              <Input id="price" type="number" placeholder="0.00" />
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="size">Size</Label>
            <Input id="size" type="number" placeholder="0.00" />
          </div>

          <Button className="w-full" variant={side === "buy" ? "default" : "destructive"}>
            Place {side === "buy" ? "Buy" : "Sell"} Order
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
