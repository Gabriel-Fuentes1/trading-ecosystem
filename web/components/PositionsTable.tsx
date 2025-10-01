"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function PositionsTable() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Open Positions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="grid grid-cols-6 gap-2 text-sm font-medium text-muted-foreground">
            <div>Symbol</div>
            <div>Side</div>
            <div className="text-right">Size</div>
            <div className="text-right">Entry</div>
            <div className="text-right">Current</div>
            <div className="text-right">P&L</div>
          </div>
          <div className="text-sm text-muted-foreground">No open positions</div>
        </div>
      </CardContent>
    </Card>
  )
}
