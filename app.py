import os
import argparse
import gradio as gr
import torch
import torchaudio
import time
import json

from datetime import datetime
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
from tortoise.utils.text import split_and_recombine_text

import music_tag

def generate(text, delimiter, emotion, prompt, voice, mic_audio, preset, seed, candidates, num_autoregressive_samples, diffusion_iterations, temperature, diffusion_sampler, breathing_room, experimentals, progress=gr.Progress()):
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
        progress(0, desc="Loading voice...")
        voice_samples, conditioning_latents = load_voice(voice)
    
    if voice_samples is not None:
        sample_voice = voice_samples[0]
        conditioning_latents = tts.get_conditioning_latents(voice_samples, progress=progress, max_chunk_size=args.cond_latent_max_chunk_size)
        torch.save(conditioning_latents, os.path.join(f'./tortoise/voices/{voice}/', f'cond_latents.pth'))
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
        'breathing_room': breathing_room,
        'progress': progress,
        'half_p': "Half Precision" in experimentals,
        'cond_free': "Conditioning-Free" in experimentals,
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
                audio_cache[f"candidate_{j}/result_{line}.wav"] = {
                    'audio': audio,
                    'text': cut_text,
                }

                os.makedirs(os.path.join(outdir, f'candidate_{j}'), exist_ok=True)
                torchaudio.save(os.path.join(outdir, f'candidate_{j}/result_{line}.wav'), audio, 24000)
        else:
            audio = gen.squeeze(0).cpu()
            audio_cache[f"result_{line}.wav"] = {
                'audio': audio,
                'text': cut_text,
            }
            torchaudio.save(os.path.join(outdir, f'result_{line}.wav'), audio, 24000)
 
    output_voice = None
    if len(texts) > 1:
        for candidate in range(candidates):
            audio_clips = []
            for line in range(len(texts)):
                if isinstance(gen, list):
                    audio = audio_cache[f'candidate_{candidate}/result_{line}.wav']['audio']
                else:
                    audio = audio_cache[f'result_{line}.wav']['audio']
                audio_clips.append(audio)
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

    info = {
        'text': text,
        'delimiter': '\\n' if delimiter == "\n" else delimiter,
        'emotion': emotion,
        'prompt': prompt,
        'voice': voice,
        'mic_audio': mic_audio,
        'preset': preset,
        'seed': seed,
        'candidates': candidates,
        'num_autoregressive_samples': num_autoregressive_samples,
        'diffusion_iterations': diffusion_iterations,
        'temperature': temperature,
        'diffusion_sampler': diffusion_sampler,
        'breathing_room': breathing_room,
        'experimentals': experimentals,
        'time': time.time()-start_time,
    }
    
    with open(os.path.join(outdir, f'input.txt'), 'w', encoding="utf-8") as f:
        f.write(json.dumps(info, indent='\t') )


    print(f"Saved to '{outdir}'")


    for path in audio_cache:
        info['text'] = audio_cache[path]['text']

        metadata = music_tag.load_file(os.path.join(outdir, path))
        metadata['lyrics'] = json.dumps(info) 
        metadata.save()
 
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

def read_metadata(file):
    j = None
    if file is not None:
        metadata = music_tag.load_file(file.name)
        if 'lyrics' in metadata:
            j = json.loads(str(metadata['lyrics']))
            print(j)
    return j

def copy_settings(file):
    metadata = read_metadata(file)
    if metadata is None:
        return None

    return (
        metadata['text'],
        metadata['delimiter'],
        metadata['emotion'],
        metadata['prompt'],
        metadata['voice'],
        metadata['mic_audio'],
        metadata['preset'],
        metadata['seed'],
        metadata['candidates'],
        metadata['num_autoregressive_samples'],
        metadata['diffusion_iterations'],
        metadata['temperature'],
        metadata['diffusion_sampler'],
        metadata['breathing_room'],
        metadata['experimentals'],
    )

def update_voices():
    return gr.Dropdown.update(choices=os.listdir(os.path.join("tortoise", "voices")) + ["microphone"])

def main():
    with gr.Blocks() as webui:
        with gr.Tab("Generate"):
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
                    refresh_voices = gr.Button(value="Refresh Voice List")
                    refresh_voices.click(update_voices,
                        inputs=None,
                        outputs=voice
                    )
                    
                    prompt.change(fn=lambda value: gr.update(value="Custom"),
                        inputs=prompt,
                        outputs=emotion
                    )
                    mic_audio.change(fn=lambda value: gr.update(value="microphone"),
                        inputs=mic_audio,
                        outputs=voice
                    )
                with gr.Column():
                    candidates = gr.Slider(value=1, minimum=1, maximum=6, step=1, label="Candidates")
                    seed = gr.Number(value=0, precision=0, label="Seed")

                    preset = gr.Radio(
                        ["Ultra Fast", "Fast", "Standard", "High Quality", "None"],
                        value="None",
                        label="Preset",
                        type="value",
                    )
                    num_autoregressive_samples = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Samples")
                    diffusion_iterations = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Iterations")

                    temperature = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.1, label="Temperature")
                    breathing_room = gr.Slider(value=12, minimum=1, maximum=32, step=1, label="Pause Size")
                    diffusion_sampler = gr.Radio(
                        ["P", "DDIM"], # + ["K_Euler_A", "DPM++2M"],
                        value="P",
                        label="Diffusion Samplers",
                        type="value",
                    )

                    experimentals = gr.CheckboxGroup(["Half Precision", "Conditioning-Free"], value=["Conditioning-Free"], label="Experimental Flags")

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
                    
                    input_settings = [
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
                        diffusion_sampler,
                        breathing_room,
                        experimentals,
                    ]

                    submit_event = submit.click(generate,
                        inputs=input_settings,
                        outputs=[selected_voice, output_audio, usedSeed],
                    )

                    #stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event])
        with gr.Tab("Utilities"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.File(type="file", label="Audio Input", file_types=["audio"])
                    copy_button = gr.Button(value="Copy Settings")
                with gr.Column():
                    metadata_out = gr.JSON(label="Audio Metadata")

                    audio_in.upload(
                        fn=read_metadata,
                        inputs=audio_in,
                        outputs=metadata_out,
                    )

                    copy_button.click(copy_settings,
                        inputs=audio_in, # JSON elements cannt be used as inputs
                        outputs=input_settings
                    )

    webui.queue().launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action='store_true', help="Lets Gradio return a public URL to use anywhere")
    parser.add_argument("--low-vram", action='store_true', help="Disables some optimizations that increases VRAM usage")
    parser.add_argument("--cond-latent-max-chunk-size", type=int, default=1000000, help="Sets an upper limit to audio chunk size when computing conditioning latents")
    args = parser.parse_args()

    print("Initializating TorToiSe...")
    tts = TextToSpeech(minor_optimizations=not args.low_vram)

    main()