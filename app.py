import os
import argparse
import gradio as gr
import torchaudio
import time
from datetime import datetime
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

VOICE_OPTIONS = [
    "random",  # special option for random voice
    "microphone",  # special option for custom voice
    "disabled",  # special option for disabled voice
]


def inference(text, emotion, prompt, voice, mic_audio, preset, seed, candidates, num_autoregressive_samples, diffusion_iterations, temperature):
    if voice != "microphone":
        voices = [voice]
    else:
        voices = []

    if emotion == "Custom" and prompt.strip() != "":
        text = f"[{prompt},] {text}"
    elif emotion != "None":
        text = f"[I am really {emotion.lower()},] {text}"

    c = None
    if voice == "microphone":
        if mic_audio is None:
            raise gr.Error("Please provide audio from mic when choosing `microphone` as a voice input")
        c = load_audio(mic_audio, 22050)


    if len(voices) == 1 or len(voices) == 0:
        if voice == "microphone":
            voice_samples, conditioning_latents = [c], None
        else:
            voice_samples, conditioning_latents = load_voice(voice)
    else:
        voice_samples, conditioning_latents = load_voices(voices)
        if voice == "microphone":
            voice_samples.extend([c])

    sample_voice = voice_samples[0] if len(voice_samples) else None

    if seed == 0:
        seed = None

    start_time = time.time()
    if preset == "none":
        gen, additionals = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset="standard",
            use_deterministic_seed=seed,
            return_deterministic_state=True,
            k=candidates,
            num_autoregressive_samples=num_autoregressive_samples,
            diffusion_iterations=diffusion_iterations,
            temperature=temperature,
        )
        seed = additionals[0]
    else:
        gen, additionals = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=preset,
            use_deterministic_seed=seed,
            return_deterministic_state=True,
            k=candidates,
            temperature=temperature,
        )
        seed = additionals[0]

    with open("results.log", "a") as f:
        f.write(
            f"{datetime.now()} | Voice: {','.join(voices)} | Text: {text} | Quality: {preset} | Time Taken (s): {time.time()-start_time} | Seed: {seed}\n"
        )

    timestamp = int(time.time())
    outdir = f"./results/{voice}/{timestamp}/"

    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, f'input.txt'), 'w') as f:
        f.write(f"{text}\n\nSeed: {seed}")

    if isinstance(gen, list):
        for j, g in enumerate(gen):
            torchaudio.save(os.path.join(outdir, f'result_{j}.wav'), g.squeeze(0).cpu(), 24000)
        return (
            (22050, sample_voice.squeeze().cpu().numpy()),
            (24000, gen[0].squeeze().cpu().numpy()),
            seed
        )
    else:
        torchaudio.save(os.path.join(outdir, f'result.wav'), gen.squeeze(0).cpu(), 24000)
        return (
            (22050, sample_voice.squeeze().cpu().numpy()),
            (24000, gen.squeeze().cpu().numpy()),
            seed
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action='store_true', help="Lets Gradio return a public URL to use anywhere")
    args = parser.parse_args()

    text = gr.Textbox(lines=4, label="Prompt")
    emotion = gr.Radio(
        ["None", "Happy", "Sad", "Angry", "Disgusted", "Arrogant", "Custom"],
        value="None",
        label="Emotion",
        type="value",
    )
    prompt = gr.Textbox(lines=1, label="Custom Emotion (if selected)")
    preset = gr.Radio(
        ["ultra_fast", "fast", "standard", "high_quality", "none"],
        value="none",
        label="Preset",
        type="value",
    )
    candidates = gr.Number(value=1, precision=0, label="Candidates")
    num_autoregressive_samples = gr.Number(value=128, precision=0, label="Samples")
    diffusion_iterations = gr.Number(value=128, precision=0, label="Iterations")
    temperature = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.1, label="Temperature")

    voice = gr.Dropdown(
        os.listdir(os.path.join("tortoise", "voices")) + VOICE_OPTIONS,
        label="Voice",
        type="value",
    )
    mic_audio = gr.Audio(
        label="Microphone Source",
        source="microphone",
        type="filepath",
    )
    seed = gr.Number(value=0, precision=0, label="Seed")

    selected_voice = gr.Audio(label="Source Sample")
    output_audio = gr.Audio(label="Output")
    usedSeed = gr.Textbox(label="Seed", placeholder="0", interactive=False) 

    interface = gr.Interface(
        fn=inference,
        inputs=[
            text,
            emotion,
            prompt,
            voice,
            mic_audio,
            preset,
            seed,
            candidates,
            num_autoregressive_samples,
            diffusion_iterations,
            temperature
        ],
        outputs=[selected_voice, output_audio, usedSeed],
        allow_flagging=False
    )
    interface.queue().launch(share=args.share)


if __name__ == "__main__":
    tts = TextToSpeech()
    main()