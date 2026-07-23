import {DataFreshness} from "./DataFreshness";
import type {VersionMeta} from "@/lib/types";

export function PageHeader({eyebrow, title, description, meta, disclaimer = "模拟研究 · 非投资建议", staleLabel}: {eyebrow:string; title:string; description:string; meta:VersionMeta; disclaimer?:string; staleLabel?:string}) {
  return <><div className="topline"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p><DataFreshness meta={meta} staleLabel={staleLabel}/></div><div className="disclaimer">{disclaimer}</div></div></>;
}
