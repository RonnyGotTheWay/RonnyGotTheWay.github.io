import type {VersionMeta} from "@/lib/types";

export function DataFreshness({meta, staleLabel = "已回退稳定快照"}: {meta: VersionMeta; staleLabel?: string}) {
  return <div className="meta">
    <span className={meta.is_stale ? "risk" : "fresh"}>{meta.is_stale ? staleLabel : "数据已校验"}</span>
    <span>截至 {meta.as_of_date}</span>
    <span>{meta.model_version}</span>
    <span>{meta.run_id}</span>
  </div>;
}
