import type React from "react"
import "./globals.css"

export const metadata = {
  title: "QuantTrade Pro - Institutional Trading Platform",
  description:
    "Advanced quantitative trading platform with AI-powered decision making, comprehensive risk management, and real-time execution.",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-white">{children}</body>
    </html>
  )
}
