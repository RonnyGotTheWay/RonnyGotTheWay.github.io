"use client";

import Link from "next/link";
import {usePathname} from "next/navigation";

const links = [
  {href: "/weekly-market", label: "市场两月报告"},
  {href: "/weekly-ranking", label: "Top 10"},
];

export function AppShell({children}: Readonly<{children: React.ReactNode}>) {
  const pathname = usePathname();

  if (pathname === "/") {
    return <main className="landing-main">{children}</main>;
  }

  return <div className="shell app-shell devil-cursor-zone">
    <aside className="sidebar">
      <Link className="brand-link" href="/" aria-label="返回 ART-RANK 封面">
        <div className="brand">ART-Rank <span>Quant</span></div>
        <div className="brand-cn">智势量选 · Research Lab</div>
      </Link>
      <nav className="nav" aria-label="研究系统导航">
        {links.map(({href, label}) => {
          const active = pathname === href;
          return <Link key={href} href={href} className={active ? "active" : undefined} aria-current={active ? "page" : undefined}>{label}</Link>;
        })}
      </nav>
      <div className="sidebar-foot">RESEARCH ONLY<br/>NO LIVE TRADING</div>
    </aside>
    <main className="content">{children}</main>
  </div>;
}
