import type {Metadata} from "next";
import {AppShell} from "@/components/AppShell";
import "./globals.css";

export const metadata: Metadata = {title: "ART-Rank Quant｜智势量选", description: "60交易日市场两月报告、板块趋势与 ART-Rank Top 10 静态研究"};

export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="zh-CN"><body><AppShell>{children}</AppShell></body></html>;
}
