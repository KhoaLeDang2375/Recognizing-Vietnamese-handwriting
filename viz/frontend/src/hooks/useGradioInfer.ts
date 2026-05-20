import { useCallback, useState } from "react";
import { infer, type InferResponse } from "@/lib/gradio";

type Status = "idle" | "loading" | "success" | "error";

type State = {
  status: Status;
  response: InferResponse | null;
  error: string | null;
};

const initial: State = { status: "idle", response: null, error: null };

export function useGradioInfer() {
  const [state, setState] = useState<State>(initial);

  const run = useCallback(
    async (file: File | Blob, model: string, usePreprocess: boolean) => {
      setState({ status: "loading", response: null, error: null });
      try {
        const response = await infer(file, model, usePreprocess);
        if (!response.ok) {
          setState({
            status: "error",
            response,
            error: response.error ?? "Inference failed.",
          });
          return response;
        }
        setState({ status: "success", response, error: null });
        return response;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setState({ status: "error", response: null, error: message });
        return null;
      }
    },
    [],
  );

  const reset = useCallback(() => setState(initial), []);

  return { ...state, run, reset };
}
