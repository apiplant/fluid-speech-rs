import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>fluid-speech-rs</IC> crate (library name <IC>fluid_speech</IC>) is the engine behind both binaries:
        model loading, the mel front end, the encoder, batched greedy TDT decoding, word timestamps and confidences.
      </Lead>

      <Section>
        <H2>Add the dependency</H2>
        <P>
          Everything beyond inference is behind features: <IC>audio</IC> (symphonia + rubato;{" "}
          <IC>fluid_speech::audio</IC>), <IC>download</IC> (ureq; fetch the model on first use), <IC>mic</IC> (cpal;{" "}
          <IC>fluid_speech::mic</IC>), <IC>cli</IC> (the <IC>fluid-speech</IC> binary) and <IC>dictate</IC> (the{" "}
          <IC>fluid-dictate</IC> binary). The defaults enable <IC>dictate</IC>, which pulls in all of them. With no
          default features the crate depends on candle, safetensors, rayon, realfft and serde only.
        </P>
        <CopyBlock command={`cargo add fluid-speech-rs --no-default-features                          # engine only, 16 kHz f32 in
cargo add fluid-speech-rs --no-default-features --features audio,download # + file decoding, model download
cargo add fluid-speech-rs --no-default-features --features cuda           # CUDA engine (needs nvcc to build)`} />
      </Section>

      <Section>
        <H2>Transcribe a file</H2>
        <Pre caption="src/main.rs" lang="rust">{`use fluid_speech::{audio, text, DeviceArg, Engine, LoadOptions};

// None: ./model, ../model, next to the binary, else downloaded (feature "download")
let engine = Engine::open(None, DeviceArg::Auto, &LoadOptions::default())?;

let samples = audio::load("talk.mp3".as_ref())?;   // any symphonia format -> 16 kHz mono (feature "audio")
let t = engine.transcribe(&samples)?;              // long audio is cut at pauses and batched

println!("{}", t.text);
print!("{}", text::srt(&t.words));
println!("confidence {:.3}", text::confidence(&t.words));`}</Pre>
        <P>
          Each <IC>Word</IC> has <IC>word</IC>, <IC>start</IC>, <IC>end</IC> (seconds) and <IC>confidence</IC> (the
          geometric mean of its tokens' probabilities).
        </P>
      </Section>

      <Section>
        <H2>Batches, threads and engines</H2>
        <Pre caption="src/main.rs" lang="rust">{`let engine = Engine::open(Some("model".into()), DeviceArg::Cuda, &LoadOptions { fast_gemm: true, ..Default::default() })?;
let many = engine.transcribe_many(&[&a, &b, &c])?;               // clips grouped by length, batched
fluid_speech::init_threads(Some(8));                             // rayon + candle pools; default: physical cores`}</Pre>
        <P>
          <IC>Engine</IC> is <IC>Sync</IC>: share one across threads, or create several. On CUDA each runs on its own
          stream with its own captured graphs.
        </P>
      </Section>
    </DocsLayout>
  );
}
