/// <reference types="vite/client" />

declare module "*.svg" {
  const content: string;
  export default content;
}

interface ImportMeta {
  readonly env: {
    readonly DEV: boolean
    readonly PROD: boolean
    readonly MODE: string
    [key: string]: any
  }
}
