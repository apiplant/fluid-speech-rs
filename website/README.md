# fluid-speech-rs website

The website for [fluid-speech-rs](https://github.com/apiplant/fluid-speech-rs). Solid 2 RC + Tailwind v4 + Vite, static build, deployed to Cloudflare.

```bash
pnpm install
pnpm dev       # local dev server
pnpm build     # -> dist/
pnpm check     # types only
```

The version shown on the site is read from `../Cargo.toml` at build time.
