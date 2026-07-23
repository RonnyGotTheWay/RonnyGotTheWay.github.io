import type {Envelope} from "./types";

const API = process.env.ART_RANK_API_URL ?? "http://127.0.0.1:8000/api/v1";
const DEFAULT_TIMEOUT = Number(process.env.NEXT_PUBLIC_FRONTEND_API_TIMEOUT_MS ?? 10_000);

export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, {...init, signal: controller.signal});
  } finally {
    clearTimeout(timer);
  }
}

export async function apiErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const body = await response.json() as {detail?: string | {message?: string}};
    if (typeof body.detail === "string") return body.detail;
    if (body.detail?.message) return body.detail.message;
  } catch {
    // The fallback is intentionally user-safe when the server body is not JSON.
  }
  return fallback;
}

export async function getApi<T>(path: string, fallback: Envelope<T>): Promise<Envelope<T>> {
  try {
    const response = await apiFetch(`${API}${path}`, {next: {revalidate: 60}});
    if (!response.ok) throw new Error(`API ${response.status}`);
    return await response.json() as Envelope<T>;
  } catch {
    return fallback;
  }
}
