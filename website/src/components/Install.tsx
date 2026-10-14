import { For, Show } from "solid-js";
import { Badge, Mono } from "./ui";
import { CopyBlock } from "./Code";
import { SITE } from "../lib/site";
import { CUDA_PLATFORM, LATEST_RELEASE_URL, PLATFORMS, assetName, downloadUrl } from "../lib/release";

const stepCard =
  "grid min-w-0 gap-5 rounded-2xl border bg-surface p-5 sm:p-6 lg:grid-cols-[minmax(14rem,0.7fr)_minmax(0,1.3fr)] lg:items-start";

function Label(props: { children: string }) {
  return <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">{props.children}</p>;
}

/** Numbered install step cards: package managers, release archives, Cargo. */
export function InstallSteps() {
  const pkg = SITE.pkg;
  const cuda = SITE.hasCuda;

  const homebrew = `brew tap apiplant/tap
brew install apiplant/tap/${pkg}`;
  const homebrewCuda = `brew install apiplant/tap/${pkg}-cuda`;
  const pacman = `curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\\n[apiplant]\\nSigLevel = Required DatabaseOptional\\nServer = https://apiplant.github.io/pacman/$arch\\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy ${pkg}`;
  const apt = `curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install ${pkg}`;
  const cargo = SITE.cargo;

  return (
    <div class="mt-8 space-y-4 sm:mt-10">
      <div class={`${stepCard} border-accent-line`}>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-mono text-xs text-accent">01</span>
            <Badge tone="accent">Recommended</Badge>
          </div>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Use a package manager</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            macOS (Apple Silicon), Arch Linux and Debian/Ubuntu (x86_64 and arm64) are all published to the
            apiplant shared repositories.
          </p>
          <Show when={cuda}>
            <p class="mt-2 text-sm leading-relaxed text-muted">
              Each has a CPU package, <Mono>{pkg}</Mono>, and a CUDA package, <Mono>{pkg}-cuda</Mono> (Linux
              x86_64 with an NVIDIA GPU). They conflict, so install one.
            </p>
          </Show>
        </div>

        <div class="min-w-0 space-y-5">
          <div>
            <Label>Homebrew</Label>
            <CopyBlock command={homebrew} />
            <Show when={cuda}>
              <div class="mt-2">
                <CopyBlock command={homebrewCuda} />
              </div>
            </Show>
          </div>

          <div>
            <Label>Arch Linux / pacman</Label>
            <CopyBlock command={pacman} />
            <Show when={cuda}>
              <p class="mt-2 text-xs text-faint">
                For CUDA, install{" "}
                <Mono>{pkg}-cuda</Mono>{" "}instead.
              </p>
            </Show>
          </div>

          <div>
            <Label>Debian / Ubuntu</Label>
            <CopyBlock command={apt} />
            <Show when={cuda}>
              <p class="mt-2 text-xs text-faint">
                For CUDA, install{" "}
                <Mono>{pkg}-cuda</Mono>{" "}instead.
              </p>
            </Show>
          </div>
        </div>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-accent">02</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Download the archive</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            One archive per platform, holding{" "}
            <For each={SITE.bins}>
              {(bin, i) => (
                <>
                  <Mono>{bin}</Mono>
                  {i() < SITE.bins.length - 1 ? ", " : ""}
                </>
              )}
            </For>{" "}
            and the README. No installation needed; unpack and run.
            <Show when={cuda}> On Linux x86_64 with an NVIDIA GPU, grab the CUDA archive instead.</Show>
          </p>
        </div>

        <ul class="min-w-0 space-y-1 border-t border-line pt-4 lg:border-t-0 lg:pt-0">
          <For each={cuda ? [...PLATFORMS, CUDA_PLATFORM] : PLATFORMS}>
            {(platform) => (
              <li class="min-w-0">
                <a
                  href={downloadUrl(platform)}
                  title={assetName(platform)}
                  class="flex min-w-0 items-baseline justify-between gap-3 rounded-md py-1 text-muted transition-colors hover:text-ink"
                >
                  <span class="shrink-0 text-sm">{platform.label}</span>
                  <span class="min-w-0 truncate font-mono text-xs text-accent">{assetName(platform)}</span>
                </a>
              </li>
            )}
          </For>
        </ul>

        <a
          href={LATEST_RELEASE_URL}
          target="_blank"
          rel="noreferrer noopener"
          class="text-sm font-medium text-accent hover:text-accent-dim lg:col-start-2"
        >
          All releases and checksums
        </a>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-faint">03</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Cargo</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">Build from source straight from the repository.</p>
        </div>

        <div class="min-w-0">
          <CopyBlock command={cargo} />
        </div>
      </div>
    </div>
  );
}
