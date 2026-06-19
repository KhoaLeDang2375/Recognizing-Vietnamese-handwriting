import { Client } from "@gradio/client";

export type InferResultItem = { text: string; conf: number };

export type InferResponse = {
  ok: boolean;
  results: InferResultItem[];
  elapsed: number;
  raw: string;
  model: string;
  preprocessed: boolean;
  processed_image: string | null;
  error: string | null;
};

/**
 * Resolve the URL of the Gradio mount.
 *
 *  - Mode A (Vite dev):  VITE_GRADIO_URL is set in .env.development
 *  - Mode B (notebook):  unset → fall back to {origin}/gradio so the SPA
 *                         talks to the FastAPI server that's serving it,
 *                         regardless of which *.gradio.live subdomain
 *                         Gradio picked.
 */
export function resolveGradioUrl(): string {
  const fromEnv = import.meta.env.VITE_GRADIO_URL;
  if (fromEnv && fromEnv.trim().length > 0) return fromEnv.replace(/\/+$/, "");
  if (typeof window !== "undefined") {
    return `${window.location.origin.replace(/\/+$/, "")}/gradio`;
  }
  return "/gradio";
}

let clientPromise: Promise<Client> | null = null;

export function getClient(): Promise<Client> {
  if (!clientPromise) {
    clientPromise = Client.connect(resolveGradioUrl());
  }
  return clientPromise;
}

/** Reset the cached client (e.g. if the backend URL changes mid-session). */
export function resetClient(): void {
  clientPromise = null;
}

const FALLBACK: InferResponse = {
  ok: false,
  results: [],
  elapsed: 0,
  raw: "",
  model: "",
  preprocessed: false,
  processed_image: null,
  error: "Empty response from backend.",
};

function isInferResponse(value: unknown): value is InferResponse {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.ok === "boolean" &&
    Array.isArray(v.results) &&
    typeof v.elapsed === "number" &&
    typeof v.raw === "string"
  );
}

export async function infer(
  imageFile: File | Blob,
  modelChoice: string,
  usePreprocess: boolean,
): Promise<InferResponse> {
  const client = await getClient();
  const result = await client.predict("/infer", [
    imageFile,
    modelChoice,
    usePreprocess,
  ]);
  // Gradio wraps the function's return in `result.data` (array of outputs).
  const payload = Array.isArray(result.data) ? result.data[0] : result.data;
  if (isInferResponse(payload)) return payload;
  return { ...FALLBACK, raw: JSON.stringify(payload ?? null, null, 2) };
}

export async function spellCheck(rawText: string): Promise<string> {
  const client = await getClient();
  const result = await client.predict("/spell_check", [rawText]);
  const payload = Array.isArray(result.data) ? result.data[0] : result.data;
  if (typeof payload === "string") {
    return payload;
  }
  throw new Error("Invalid response from spell check API: " + JSON.stringify(payload));
}

