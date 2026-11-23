import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { Mono } from "../ui";
import { CopyBlock } from "../Code";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        fluid-speech-rs supports{" "}
        <a
          href="https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-coreml"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          Fluid Speech Mini
        </a>
        {" "}and{" "}
        <a
          href="https://huggingface.co/FluidVoice-Labs/fluid-speech-pro"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          Fluid Speech Pro
        </a>
        . Mini is FluidVoice's English dictation model; Pro is its multilingual Transformers checkpoint. Both run
        through a Rust (<Mono>candle</Mono>) engine with hand-written CUDA kernels. The project ships as a library,
        a transcriber and a Linux and macOS dictation daemon.
      </Lead>

      <Section>
        <H2>Models</H2>
        <P>
          <IC>Mini</IC> is the compact English model: a 327 MB safetensors package that keeps the Core ML
          quantisation and expands to fp16 at load. Mini is the fallback when no Pro or other local model is available.
          The model weights are CC BY 4.0.
        </P>
        <P>
          <IC>Pro</IC> is the Transformers checkpoint for 25 European languages. It is about 2.5 GB and can be loaded directly;
          there is no Python conversion step. A local copy can live at <IC>model/fluid-speech-pro/</IC> alongside
          Mini.
        </P>
        <CopyBlock command={`# choose a model explicitly; missing models download on demand
fluid-speech recording.wav --model mini
fluid-speech recording.wav --model pro

# or download Pro into the project model folder
hf download FluidVoice-Labs/fluid-speech-pro config.json processor_config.json tokenizer.json model.safetensors LICENSE NOTICE --local-dir model/fluid-speech-pro`} />
      </Section>

      <Section>
        <H2>The binaries</H2>
        <UL>
          <LI>
            <a href="/docs/cli" class="text-accent hover:text-accent-dim">
              <IC>fluid-speech</IC>
            </a>{" "}
            transcribes audio files or the microphone to text, JSON or SRT.
          </LI>
          <LI>
            <a href="/docs/dictate" class="text-accent hover:text-accent-dim">
              <IC>fluid-dictate</IC>
            </a>{" "}
            is a push-to-talk dictation daemon for Linux and macOS.
          </LI>
        </UL>
      </Section>

      <Section>
        <H2>Model resolution</H2>
        <P>
          Pass <IC>--model mini</IC> or <IC>--model pro</IC> to choose explicitly. Each option uses a local or cached
          copy and downloads that model if needed. You can also pass a local model directory; the Pro Hugging Face ID{" "}
          <IC>FluidVoice-Labs/fluid-speech-pro</IC> is accepted as an alias for Pro. <IC>FLUID_MODEL</IC> sets the
          same option.
        </P>
        <P>With no model selection, resolution follows this order:</P>
        <UL>
          <LI>
            a local Pro at <IC>model/fluid-speech-pro/</IC>, or a cached Pro at{" "}
            <IC>$XDG_CACHE_HOME/fluid-speech-rs/fluid-speech-pro/</IC>
          </LI>
          <LI>
            local <IC>model/</IC> directories near the current directory or binary
          </LI>
          <LI>
            cached Mini, or Mini downloaded from Hugging Face into <IC>$XDG_CACHE_HOME/fluid-speech-rs/</IC> (by
            default <IC>~/.cache/fluid-speech-rs/</IC>)
          </LI>
        </UL>
      </Section>

      <Section>
        <H2>Speed and accuracy</H2>
        <P>RTX 4090 + Ryzen 9 7950X, release build, audio already in memory.</P>
        <FlagTable
          rows={[
            { flag: "2 s clip, CUDA", meaning: "3.2 ms end to end (mel, encoder, decoder)" },
            { flag: "11 s clip, CUDA", meaning: "5.3 ms" },
            { flag: "LibriSpeech test-clean, CUDA", meaning: "5.4 h in 4.2 s, 4,600x real time (5,800x with --fast)" },
            { flag: "11 s clip, CPU, 32 threads", meaning: "16x real time" },
            { flag: "WER, test-clean", meaning: "2.07 % (1.82 % with the Whisper normaliser)" },
          ]}
        />
        <P>
          On a 328-clip subset the official ONNX build scores 1.95 % and this engine 1.96 %, one word apart.
        </P>
      </Section>
    </DocsLayout>
  );
}
