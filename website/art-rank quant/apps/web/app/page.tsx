import type {Metadata} from "next";
import Image from "next/image";
import Link from "next/link";

export const metadata: Metadata = {
  title: "ART-RANK｜量化研究系统",
  description: "汇总最近60个交易日市场变化、板块趋势与 ART-Rank 模型选股的两月研究终端",
};

export default function Home() {
  return <section className="landing-page">
    <Image className="landing-cover" src="/images/art-rank-cover.webp" alt="抽象的翡翠色量化信号与金色数据轨迹" fill priority sizes="100vw"/>
    <div className="landing-shade" aria-hidden="true"/>
    <div className="landing-grid" aria-hidden="true"/>
    <div className="landing-brand" aria-label="ART-RANK Quant Research System">
      <span className="landing-mark" aria-hidden="true">AR</span>
      <span>ART‑RANK</span>
    </div>
    <div className="landing-copy">
      <div className="landing-eyebrow"><span/> QUANT RESEARCH SYSTEM</div>
      <h1>从市场噪声中，<br/><em>识别下一步秩序</em></h1>
      <p>汇总最近60个交易日市场变化、板块趋势与 ART-Rank 模型选股的静态两月研究终端</p>
      <Link className="enter-button" href="/weekly-market">
        <span>进入 ART‑RANK</span>
        <i aria-hidden="true">↗</i>
      </Link>
      <div className="landing-features" aria-label="系统功能">
        <span>01 市场两月报告</span>
        <span>02 Top 10</span>
        <span>03 板块研判</span>
        <span>04 静态发布</span>
      </div>
    </div>
    <div className="landing-foot">
      <span>Research only · No live trading</span>
      <span className="landing-status"><i/> System ready</span>
    </div>
  </section>;
}
