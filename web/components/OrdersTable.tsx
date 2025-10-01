"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function OrdersTable() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Active Orders</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="grid grid-cols-6 gap-2 text-sm font-medium text-muted-foreground">
            <div>Symbol</div>
            <div>Type</div>
            <div>Side</div>
            <div className="text-right">Price</div>
            <div className="text-right">Size</div>
            <div>Status</div>
          </div>
          <div className="text-sm text-muted-foreground">No active orders</div>
        </div>
      </CardContent>
    </Card>
  )
}
