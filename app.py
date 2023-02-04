import os
import argparse
import gradio as gr
import torch
import torchaudio
import time

from datetime import datetime
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

def inference(text, emotion, prompt, voice, mic_audio, preset, seed, candidates, num_autoregressive_samples, diffusion_iterations, temperature, progress=gr.Progress()):
    if voice != "microphone":
        voices = [voice]
    else:
        voices = []

    if emotion == "Custom" and prompt.strip() != "":
        text = f"[{prompt},] {text}"
    elif emotion != "None":
        text = f"[I am really {emotion.lower()},] {text}"

    if voice == "microphone":
        if mic_audio is None:
            raise gr.Error("Please provide audio from mic when choosing `microphone` as a voice input")
        mic = load_audio(mic_audio, 22050)
        voice_samples, conditioning_latents = [mic], None
    else:
        voice_samples, conditioning_latents = load_voice(voice)
    
    if voice_samples is not None:
        sample_voice = voice_samples[0]
        conditioning_latents = tts.get_conditioning_latents(voice_samples)
        torch.save(conditioning_latents, os.path.join(f'./tortoise/voices/{voice}/', f'latents.pth'))
        voice_samples = None
    else:
        sample_voice = None

    if seed == 0:
        seed = None

    start_time = time.time()

    presets = {
        'ultra_fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
        'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
        'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
        'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
        'none': {'num_autoregressive_samples': num_autoregressive_samples, 'diffusion_iterations': diffusion_iterations},
    }
    settings = {
        'temperature': temperature, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
        'top_p': .8,
        'cond_free_k': 2.0, 'diffusion_temperature': 1.0,

        'voice_samples': voice_samples,
        'conditioning_latents': conditioning_latents,
        'use_deterministic_seed': seed,
        'return_deterministic_state': True,
        'k': candidates,
        'progress': progress,
    }
    settings.update(presets[preset])
    gen, additionals = tts.tts( text, **settings )
    seed = additionals[0]

    info = f"{datetime.now()} | Voice: {','.join(voices)} | Text: {text} | Quality: {preset} preset / {num_autoregressive_samples} samples / {diffusion_iterations} iterations | Temperature: {temperature} | Time Taken (s): {time.time()-start_time} | Seed: {seed}\n"
    with open("results.log", "a") as f:
        f.write(info)

    timestamp = int(time.time())
    outdir = f"./results/{voice}/{timestamp}/"

    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, f'input.txt'), 'w') as f:
        f.write(f"{text}\n\n{info}")

    if isinstance(gen, list):
        for j, g in enumerate(gen):
            torchaudio.save(os.path.join(outdir, f'result_{j}.wav'), g.squeeze(0).cpu(), 24000)
        
        output_voice = gen[0]
    else:
        torchaudio.save(os.path.join(outdir, f'result.wav'), gen.squeeze(0).cpu(), 24000)
        output_voice = gen

    output_voice = (24000, output_voice.squeeze().cpu().numpy())

    if sample_voice is not None:
        sample_voice = (22050, sample_voice.squeeze().cpu().numpy())

    return (
        sample_voice,
        output_voice,
        seed
    )

def main():
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
    candidates = gr.Slider(value=1, minimum=1, maximum=6, label="Candidates")
    num_autoregressive_samples = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Samples")
    diffusion_iterations = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Iterations")
    temperature = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.1, label="Temperature")

    voice = gr.Dropdown(
        os.listdir(os.path.join("tortoise", "voices")) + ["random", "microphone", "disabled"],
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
        allow_flagging='never'
    )
    interface.queue().launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action='store_true', help="Lets Gradio return a public URL to use anywhere")
    parser.add_argument("--low-vram", action='store_true', help="Disables some optimizations that increases VRAM usage")
    args = parser.parse_args()

    tts = TextToSpeech(minor_optimizations=not args.low_vram)

    main()