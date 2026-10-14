# Generated from packaging/homebrew/fluid-speech-rs.rb in apiplant/fluid-speech-rs by the
# release workflow, which fills in the version and checksums and commits the
# result to apiplant/homebrew-tap as Formula/fluid-speech-rs.rb. Changes belong in
# the source repository: the next release overwrites this file.
class FluidSpeechRs < Formula
  desc "Fluid Speech Mini speech-to-text CLI and dictation daemon (candle)"
  homepage "https://github.com/apiplant/fluid-speech-rs"
  version "@VERSION@"
  license "Apache-2.0"

  # No bottles: the release archives *are* the binaries, so the formula only
  # unpacks what the tagged workflow already built for each platform.
  on_macos do
    on_arm do
      url "https://github.com/apiplant/fluid-speech-rs/releases/download/v@VERSION@/fluid-speech-rs-v@VERSION@-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA_MACOS_ARM64@"
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/apiplant/fluid-speech-rs/releases/download/v@VERSION@/fluid-speech-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_X86_64@"
    end
    on_arm do
      url "https://github.com/apiplant/fluid-speech-rs/releases/download/v@VERSION@/fluid-speech-rs-v@VERSION@-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_ARM64@"
    end
  end

  conflicts_with "fluid-speech-rs-cuda", because: "both install the same binaries"

  def install
    bin.install "fluid-speech"
    bin.install "fluid-dictate"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/fluid-speech --version")
  end
end
