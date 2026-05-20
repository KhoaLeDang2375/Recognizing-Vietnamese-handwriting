/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_GRADIO_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
