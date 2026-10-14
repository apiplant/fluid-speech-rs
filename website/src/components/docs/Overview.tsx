import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { Mono } from "../ui";
import { CopyBlock } from "../Code";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        fluid-speech-rs runs{" "}
        <a
          href="https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-coreml"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          Fluid Speech Mini
        </a>
        , FluidVoice's English dictation model (a Parakeet TDT 0.6B), outside Apple platforms: the Core ML package is
        ported to safetensors and executed by a Rust (<Mono>candle</Mono>) engine with hand-written CUDA kernels. It
        ships as a library crate, a transcriber and a Linux and macOS dictation daemon.
      </Lead>

      <Section>
        <H2>The model</H2>
        <P>
          A FastConformer encoder (24 layers, d_model 1024), a 2-layer LSTM prediction network and a TDT joint (8,192
          tokens + blank, durations 0 to 4). The safetensors file keeps the Core ML quantisation: 2/3/4-bit palettized
          encoder linears and int8 everywhere else, 327 MB. It expands to fp16 (1.25 GB) in about 0.1 s at load,
          bit-identical to a dense export. Weights are CC BY 4.0.
        </P>
        <CopyBlock command={`# optional: the binaries download it on first use
hf download apiplant/fluid-speech-mini-safetensors --local-dir model`} />
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
        <P>Both binaries look for the model in this order:</P>
        <UL>
          <LI>
            <IC>--model DIR</IC> (a directory with <IC>model.safetensors</IC>, <IC>config.json</IC> and{" "}
            <IC>vocab.json</IC>)
          </LI>
          <LI>
            the <IC>FLUID_MODEL</IC> environment variable
          </LI>
          <LI>
            <IC>./model</IC>, <IC>../model</IC>, or a <IC>model</IC> directory next to the binary
          </LI>
          <LI>
            otherwise it is downloaded from Hugging Face on first use (327 MB) into{" "}
            <IC>$XDG_CACHE_HOME/fluid-speech-rs/</IC> (<IC>~/.cache/fluid-speech-rs/</IC> by default) and reused
            afterwards. Files are fetched to a <IC>.part</IC> file and renamed when complete, so an interrupted
            download never leaves a file that looks done.
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
