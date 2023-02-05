import os
import argparse
import gradio as gr
import torch
import torchaudio
import time

from datetime import datetime
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
from tortoise.utils.text import split_and_recombine_text

def generate(text, delimiter, emotion, prompt, voice, mic_audio, preset, seed, candidates, num_autoregressive_samples, diffusion_iterations, temperature, diffusion_sampler, progress=gr.Progress()):
    if voice != "microphone":
        voices = [voice]
    else:
        voices = []

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

    settings = {
        'temperature': temperature, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
        'top_p': .8,
        'cond_free_k': 2.0, 'diffusion_temperature': 1.0,

        'num_autoregressive_samples': num_autoregressive_samples,
        'diffusion_iterations': diffusion_iterations,

        'voice_samples': voice_samples,
        'conditioning_latents': conditioning_latents,
        'use_deterministic_seed': seed,
        'return_deterministic_state': True,
        'k': candidates,
        'diffusion_sampler': diffusion_sampler,
        'progress': progress,
    }

    if delimiter == "\\n":
        delimiter = "\n"

    if delimiter != "" and delimiter in text:
        texts = text.split(delimiter)
    else:
        texts = split_and_recombine_text(text)
 
 
    timestamp = int(time.time())
    outdir = f"./results/{voice}/{timestamp}/"
 
    os.makedirs(outdir, exist_ok=True)
 

    audio_cache = {}
    for line, cut_text in enumerate(texts):
        if emotion == "Custom" and prompt.strip() != "":
            cut_text = f"[{prompt},] {cut_text}"
        elif emotion != "None":
            cut_text = f"[I am really {emotion.lower()},] {cut_text}"

        print(f"[{str(line+1)}/{str(len(texts))}] Generating line: {cut_text}")

        gen, additionals = tts.tts(cut_text, **settings )
        seed = additionals[0]
 
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                audio = g.squeeze(0).cpu()
                audio_cache[f"candidate_{j}/result_{line}.wav"] = audio

                os.makedirs(os.path.join(outdir, f'candidate_{j}'), exist_ok=True)
                torchaudio.save(os.path.join(outdir, f'candidate_{j}/result_{line}.wav'), audio, 24000)
        else:
            audio = gen.squeeze(0).cpu()
            audio_cache[f"result_{line}.wav"] = audio
            torchaudio.save(os.path.join(outdir, f'result_{line}.wav'), audio, 24000)
 
    output_voice = None
    if len(texts) > 1:
        for candidate in range(candidates):
            audio_clips = []
            for line in range(len(texts)):
                if isinstance(gen, list):
                    piece = audio_cache[f'candidate_{candidate}/result_{line}.wav']
                else:
                    piece = audio_cache[f'result_{line}.wav']
                audio_clips.append(piece)
            audio_clips = torch.cat(audio_clips, dim=-1)
            torchaudio.save(os.path.join(outdir, f'combined_{candidate}.wav'), audio_clips, 24000)
            
            if output_voice is None:
                output_voice = (24000, audio_clips.squeeze().cpu().numpy())
    else:
        if isinstance(gen, list):
            output_voice = gen[0]
        else:
            output_voice = gen
        output_voice = (24000, output_voice.squeeze().cpu().numpy())
 
    info = f"{datetime.now()} | Voice: {','.join(voices)} | Text: {text} | Quality: {preset} preset / {num_autoregressive_samples} samples / {diffusion_iterations} iterations | Temperature: {temperature} | Time Taken (s): {time.time()-start_time} | Seed: {seed}\n"
    
    with open(os.path.join(outdir, f'input.txt'), 'w', encoding="utf-8") as f:
        f.write(info)

    with open("results.log", "w", encoding="utf-8") as f:
        f.write(info)

    print(f"Saved to '{outdir}'")
    
 
    if sample_voice is not None:
        sample_voice = (22050, sample_voice.squeeze().cpu().numpy())
 
    audio_clips = []
    return (
        sample_voice,
        output_voice, 
        seed
    )

def update_presets(value):
    PRESETS = {
        'Ultra Fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
        'Fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
        'Standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
        'High Quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
    }
    
    if value in PRESETS:
        preset = PRESETS[value]
        return (gr.update(value=preset['num_autoregressive_samples']), gr.update(value=preset['diffusion_iterations']))
    else:
        return (gr.update(), gr.update())

def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(lines=4, label="Prompt")
                delimiter = gr.Textbox(lines=1, label="Line Delimiter", placeholder="\\n")

                emotion = gr.Radio(
                    ["None", "Happy", "Sad", "Angry", "Disgusted", "Arrogant", "Custom"],
                    value="None",
                    label="Emotion",
                    type="value",
                    interactive=True
                )
                prompt = gr.Textbox(lines=1, label="Custom Emotion + Prompt (if selected)")
                voice = gr.Dropdown(
                    os.listdir(os.path.join("tortoise", "voices")) + ["microphone"],
                    label="Voice",
                    type="value",
                )
                mic_audio = gr.Audio(
                    label="Microphone Source",
                    source="microphone",
                    type="filepath",
                )
                
                candidates = gr.Slider(value=1, minimum=1, maximum=6, step=1, label="Candidates")
                seed = gr.Number(value=0, precision=0, label="Seed")

                preset = gr.Radio(
                    ["Ultra Fast", "Fast", "Standard", "High Quality", "None"],
                    value="None",
                    label="Preset",
                    type="value",
                )
                num_autoregressive_samples = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Samples", interactive=True)
                diffusion_iterations = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Iterations", interactive=True)

                temperature = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.1, label="Temperature")
                diffusion_sampler = gr.Radio(
                    ["P", "DDIM"],
                    value="P",
                    label="Diffusion Samplers",
                    type="value",
                )

                prompt.change(fn=lambda value: gr.update(value="Custom"),
                    inputs=prompt,
                    outputs=emotion
                )
                mic_audio.change(fn=lambda value: gr.update(value="microphone"),
                    inputs=mic_audio,
                    outputs=voice
                )

                preset.change(fn=update_presets,
                    inputs=preset,
                    outputs=[
                        num_autoregressive_samples,
                        diffusion_iterations,
                    ],
                )
            with gr.Column():
                selected_voice = gr.Audio(label="Source Sample")
                output_audio = gr.Audio(label="Output")
                usedSeed = gr.Textbox(label="Seed", placeholder="0", interactive=False) 
                
                submit = gr.Button(value="Generate")
                #stop = gr.Button(value="Stop")
                
                submit_event = submit.click(generate,
                    inputs=[
                        text,
                        delimiter,
                        emotion,
                        prompt,
                        voice,
                        mic_audio,
                        preset,
                        seed,
                        candidates,
                        num_autoregressive_samples,
                        diffusion_iterations,
                        temperature,
                        diffusion_sampler
                    ],
                    outputs=[selected_voice, output_audio, usedSeed],
                )

                #stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event])

    demo.queue().launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action='store_true', help="Lets Gradio return a public URL to use anywhere")
    parser.add_argument("--low-vram", action='store_true', help="Disables some optimizations that increases VRAM usage")
    args = parser.parse_args()

    tts = TextToSpeech(minor_optimizations=not args.low_vram)

    main()