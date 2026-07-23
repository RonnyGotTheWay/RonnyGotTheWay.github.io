import {readFile} from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

export async function GET() {
  const candidates = [
    path.resolve(/*turbopackIgnore: true*/ process.cwd(), "artifacts/published/weekly_reports/stable.json"),
    path.resolve(/*turbopackIgnore: true*/ process.cwd(), "../../artifacts/published/weekly_reports/stable.json"),
  ];
  for (const candidate of candidates) {
    try {
      const body = await readFile(candidate, "utf8");
      return new Response(body, {headers: {"content-type": "application/json; charset=utf-8"}});
    } catch {
      // Try the next workspace-relative location.
    }
  }
  try {
    const response = await fetch("http://127.0.0.1:8000/api/v1/reports/weekly/latest", {cache: "no-store"});
    if (response.ok) return new Response(await response.text(), {headers: {"content-type": "application/json"}});
  } catch {
    // The user-facing response below explains the recovery command.
  }
  return Response.json({detail: "尚无本地周报，请运行 make weekly-report。"}, {status: 404});
}
