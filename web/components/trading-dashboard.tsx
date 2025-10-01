"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { TradingChart } from "./trading-chart"
import { OrderBook } from "./order-book"
import { PositionsTable } from "./positions-table"
import { OrdersTable } from "./orders-table"
import { RiskMetrics } from "./risk-metrics"
import { MarketData } from "./market-data"
import { TradingPanel } from "./trading-panel"
import { useWebSocket } from "@/hooks/use-websocket"
import { useDashboardData } from "@/hooks/use-dashboard-data"
import { TrendingUp, TrendingDown, DollarSign, AlertTriangle, Settings, LogOut } from "lucide-react"

export function TradingDashboard() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT")
  const { data: dashboardData, loading, error } = useDashboardData()
  const { marketData, isConnected } = useWebSocket()

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="w-96">
          <CardContent className="pt-6">
            <div className="text-center">
              <AlertTriangle className="h-12 w-12 text-destructive mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">{"Connection Error"}</h3>
              <p className="text-muted-foreground mb-4">{"Unable to load dashboard data"}</p>
              <Button onClick={() => window.location.reload()}>{"Retry"}</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="trading-grid">
      {/* Header */}
      <header className="trading-header flex items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-bold">{"QuantTrade Pro"}</h1>
          <Badge variant={isConnected ? "default" : "destructive"} className="text-xs">
            {isConnected ? "LIVE" : "DISCONNECTED"}
          </Badge>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-sm text-muted-foreground">{new Date().toLocaleTimeString()}</div>
          <Button variant="ghost" size="sm">
            <Settings className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm">
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Sidebar */}
      <aside className="trading-sidebar p-4 space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-muted-foreground mb-3 uppercase tracking-wide">
            {"Portfolio Overview"}
          </h3>
          <div className="space-y-3">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">{"Total Value"}</p>
                    <p className="text-2xl font-bold">${dashboardData?.portfolio_value?.toLocaleString() || "0"}</p>
                  </div>
                  <DollarSign className="h-8 w-8 text-primary" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">{"Daily P&L"}</p>
                    <p
                      className={`text-xl font-bold ${
                        (dashboardData?.daily_pnl || 0) >= 0 ? "text-success" : "text-destructive"
                      }`}
                    >
                      ${dashboardData?.daily_pnl?.toLocaleString() || "0"}
                    </p>
                    <p
                      className={`text-xs ${
                        (dashboardData?.daily_pnl_percentage || 0) >= 0 ? "text-success" : "text-destructive"
                      }`}
                    >
                      {(dashboardData?.daily_pnl_percentage || 0) >= 0 ? "+" : ""}
                      {dashboardData?.daily_pnl_percentage?.toFixed(2) || "0"}%
                    </p>
                  </div>
                  {(dashboardData?.daily_pnl || 0) >= 0 ? (
                    <TrendingUp className="h-8 w-8 text-success" />
                  ) : (
                    <TrendingDown className="h-8 w-8 text-destructive" />
                  )}
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-2 gap-2">
              <Card>
                <CardContent className="p-3">
                  <p className="text-xs text-muted-foreground">{"Positions"}</p>
                  <p className="text-lg font-bold">{dashboardData?.total_positions || 0}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3">
                  <p className="text-xs text-muted-foreground">{"Orders"}</p>
                  <p className="text-lg font-bold">{dashboardData?.active_orders || 0}</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>

        <MarketData marketData={marketData} onSymbolSelect={setSelectedSymbol} />
      </aside>

      {/* Main Content */}
      <main className="trading-main p-4 space-y-4">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-full">
          <div className="lg:col-span-2 space-y-4">
            <TradingChart symbol={selectedSymbol} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <PositionsTable positions={dashboardData?.top_performers || []} />
              <OrdersTable orders={dashboardData?.recent_trades || []} />
            </div>
          </div>
          <div className="space-y-4">
            <RiskMetrics metrics={dashboardData?.risk_metrics || {}} />
            <OrderBook symbol={selectedSymbol} />
          </div>
        </div>
      </main>

      {/* Right Panel */}
      <aside className="trading-panel p-4">
        <TradingPanel symbol={selectedSymbol} />
      </aside>
    </div>
  )
}
