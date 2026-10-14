import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1>fluid-speech</H1>
      <Lead>
        Transcribe audio files or the microphone. Files can be in any format symphonia decodes (wav, flac,
        mp3, ogg/vorbis, aac/m4a); they are mixed to mono and resampled to 16 kHz.
      </Lead>

      <Section>
        <H2>Examples</H2>
        <CopyBlock
          command={`fluid-speech recording.wav                         # CUDA if built with it and a GPU is present, else CPU
fluid-speech talk.mp3 --device cpu --format srt > talk.srt
fluid-speech a.flac b.m4a --format json            # words with start/end times and confidence
fluid-speech hour.flac --bench                     # timings + real-time factor on stderr
fluid-speech --manifest clips.jsonl --out hyps.jsonl  # {"id","audio"} lines in, {"id","hyp"} out

fluid-speech --mic                                 # live from the default input; Ctrl-C to stop
fluid-speech --list-mics
fluid-speech --mic --mic-device wave --format json # one JSON line per utterance`}
        />
      </Section>

      <Section>
        <H2>Options</H2>
        <FlagTable
          rows={[
            { flag: "-m, --model DIR", meaning: <>A directory with <IC>model.safetensors</IC>, <IC>config.json</IC> and <IC>vocab.json</IC>. Also <IC>FLUID_MODEL</IC>. Default: found or downloaded.</> },
            { flag: "-d, --device auto|cpu|cuda", meaning: "Where to run. auto picks CUDA when the build has it and a GPU is present." },
            { flag: "-f, --format text|json|srt", meaning: "Output format. json carries words with start/end times and confidences." },
            { flag: "--max-pass 40", meaning: "Clips up to this many seconds are encoded in one pass; longer audio is cut at quiet points into ~30 s pieces. In --mic mode, the longest single utterance." },
            { flag: "--batch-size 16", meaning: "Pieces per encoder batch." },
            { flag: "--threads N", meaning: "Defaults to the physical core count." },
            { flag: "--fast", meaning: "CUDA: fp16 accumulation in the encoder GEMMs (~20% faster on GeForce, same WER)." },
            { flag: "--tune", meaning: "CUDA: autotune cuBLASLt algorithms per GEMM shape on first use." },
            { flag: "--bench, --warmup, --repeat N", meaning: "Print timings and the real-time factor on stderr; median of N runs." },
          ]}
        />
      </Section>

      <Section>
        <H2>Microphone mode</H2>
        <P>
          <IC>--mic</IC> captures the chosen input and splits it into utterances with an energy VAD on the 200 Hz high-passed signal. The
          speech threshold is 4x a running noise-floor estimate, or <IC>--vad-threshold</IC> if you set one.
        </P>
        <UL>
          <LI>While you speak, the current utterance is re-transcribed about every half second and shown dimmed on stderr.</LI>
          <LI>After a pause of <IC>--silence-ms</IC> (700 ms by default) the final text goes to stdout: plain text, one JSON line per utterance, or SRT cues.</LI>
          <LI>Ctrl-C flushes the utterance in progress and exits.</LI>
        </UL>
        <FlagTable
          rows={[
            { flag: "--mic", meaning: "Transcribe the microphone live." },
            { flag: "--mic-device NAME", meaning: "Any part of a name from --list-mics. Defaults to the system default input." },
            { flag: "--list-mics", meaning: "List audio input devices (* marks the default) and exit." },
            { flag: "--silence-ms 700", meaning: "Pause that ends an utterance." },
            { flag: "--vad-threshold X", meaning: "Fixed RMS level (of the 200 Hz high-passed signal) counted as speech. Default: adaptive." },
            { flag: "--min-confidence 0", meaning: "Drop one- or two-word utterances below this confidence. Off by default: this model decodes noise to nothing, while short real phrases often score 0.4 to 0.9." },
          ]}
        />
        <P>
          On Linux the microphone is opened through PulseAudio's protocol, which PipeWire also serves via
          pipewire-pulse, so <IC>--mic-device</IC> picks the sound server's named sources. Plain ALSA is the
          fallback when no sound server runs.
        </P>
      </Section>
    </DocsLayout>
  );
}
