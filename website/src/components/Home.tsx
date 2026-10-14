import { For, Show } from "solid-js";
import { Badge, LinkButton } from "./ui";
import { CopyBlock } from "./Code";
import { Pre } from "./docs/Prose";
import { InstallSteps } from "./Install";
import { highlight, type HighlightLang } from "../lib/highlight";
import { GITHUB_URL } from "../lib/links";
import { SITE } from "../lib/site";

/* A fake terminal panel showing a real invocation and its output. */
function TerminalDemo() {
  const t = SITE.terminal;
  return (
    <div class="min-w-0 overflow-hidden rounded-xl border border-line shadow-2xl">
      <div class="flex items-center gap-2 border-b border-line bg-surface px-4 py-2.5">
        <span class="h-2.5 w-2.5 rounded-full bg-danger" />
        <span class="h-2.5 w-2.5 rounded-full bg-warn" />
        <span class="h-2.5 w-2.5 rounded-full bg-success" />
        <span class="ml-2 font-mono text-xs text-faint">{t.title}</span>
      </div>
      <pre class="overflow-x-auto bg-code-bg px-4 py-4 font-mono text-[0.78rem] leading-relaxed">
        <code class="language-bash">
          <span class="select-none text-faint">$ </span>
          <span innerHTML={highlight(t.command, "bash")} />
        </code>
        {"\n\n"}
        <code class={`language-${t.outputLang}`}>
          <span innerHTML={highlight(t.output, t.outputLang as HighlightLang)} />
        </code>
      </pre>
    </div>
  );
}

function Cards() {
  return (
    <div class="grid gap-4 sm:grid-cols-2">
      <For each={SITE.cards}>
        {(b) => (
          <a
            href={b.href}
            class="group block rounded-xl border border-line bg-surface p-5 transition-colors hover:border-line-strong"
          >
            <div class="flex items-center justify-between gap-2">
              <h3 class="font-mono text-[0.9375rem] font-semibold tracking-tight text-ink">{b.name}</h3>
              <span class="text-xs text-faint transition-colors group-hover:text-accent">docs →</span>
            </div>
            <p class="mt-1 text-xs font-medium uppercase tracking-[0.1em] text-accent">{b.tagline}</p>
            <p class="mt-2.5 text-sm leading-relaxed text-muted">{b.body}</p>
          </a>
        )}
      </For>
    </div>
  );
}

function Features() {
  return (
    <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <For each={SITE.features}>
        {(f) => (
          <div class="rounded-xl border border-line bg-surface p-5">
            <h3 class="text-[0.9375rem] font-semibold tracking-tight text-ink">{f.title}</h3>
            <p class="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
          </div>
        )}
      </For>
    </div>
  );
}

export function Home() {
  return (
    <div class="mx-auto w-full max-w-6xl px-5">
      {/* Hero */}
      <section class="grid items-center gap-10 py-16 sm:py-20 lg:grid-cols-2 lg:gap-12">
        <div class="min-w-0">
          <div class="flex flex-wrap items-center gap-2">
            <Badge tone="accent">v{__VERSION__}</Badge>
            <For each={SITE.badges}>{(b) => <Badge>{b}</Badge>}</For>
          </div>
          <h1 class="mt-5 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
            {SITE.hero.pre}
            <span class="text-accent">{SITE.hero.accent}</span>
            {SITE.hero.post}
          </h1>
          <p class="mt-4 max-w-lg text-lg leading-relaxed text-muted">{SITE.lead}</p>
          <Show when={SITE.demo}>
            {(demo) => (
              <div class="mt-7">
                <LinkButton
                  href={demo().href}
                  variant="primary"
                  class="!px-8 !py-4 !text-lg shadow-lg shadow-accent/20"
                >
                  {demo().label}
                </LinkButton>
              </div>
            )}
          </Show>
          <div class="mt-4 flex flex-wrap gap-3">
            <LinkButton href={GITHUB_URL} size="lg" variant={SITE.demo ? "secondary" : "primary"}>
              View on GitHub
            </LinkButton>
            <LinkButton href="/docs" size="lg">
              Read the docs
            </LinkButton>
            <LinkButton href="/#install" size="lg">
              Install
            </LinkButton>
          </div>
          <p class="mt-5 text-sm text-faint">
            {SITE.heroNote}
          </p>
        </div>
        <TerminalDemo />
      </section>

      {/* Cards */}
      <section class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">{SITE.cardsTitle}</h2>
        <p class="mt-2 max-w-2xl text-muted">{SITE.cardsLead}</p>
        <div class="mt-8">
          <Cards />
        </div>
      </section>

      {/* Features */}
      <section id="features" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">{SITE.featuresTitle}</h2>
        <p class="mt-2 max-w-2xl text-muted">{SITE.featuresLead}</p>
        <div class="mt-8">
          <Features />
        </div>
      </section>

      {/* Install */}
      <section id="install" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Install</h2>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          Use Homebrew, pacman or apt when your platform has it. Otherwise take the prebuilt archive, or build
          from source.
        </p>
        <InstallSteps />
      </section>

      {/* Library */}
      <section class="border-t border-line pb-20 pt-16">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h2 class="text-2xl font-semibold tracking-tight text-ink">As a library</h2>
          <LinkButton href="/docs/library" size="sm">
            Full library docs →
          </LinkButton>
        </div>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">{SITE.lib.lead}</p>
        <div class="mt-6">
          <CopyBlock command={SITE.lib.add} />
        </div>
        <Pre caption={SITE.lib.caption} lang="rust">
          {SITE.lib.snippet}
        </Pre>
      </section>
    </div>
  );
}
