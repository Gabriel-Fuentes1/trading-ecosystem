"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function RiskMetrics() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Risk Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-muted-foreground">Portfolio Value</div>
            <div className="text-2xl font-bold">$0.00</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">Available Balance</div>
            <div className="text-2xl font-bold">$0.00</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">Total P&L</div>
            <div className="text-2xl font-bold text-green-600">$0.00</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">Win Rate</div>
            <div className="text-2xl font-bold">0%</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
