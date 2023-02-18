import os
import webui as mrq

print('DEPRECATION WARNING: this repo has been refractored to focus entirely on tortoise-tts. Please migrate to https://git.ecker.tech/mrq/ai-voice-cloning if you seek new features.')

if 'TORTOISE_MODELS_DIR' not in os.environ:
    os.environ['TORTOISE_MODELS_DIR'] = os.path.realpath(os.path.join(os.getcwd(), './models/tortoise/'))

if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = os.path.realpath(os.path.join(os.getcwd(), './models/transformers/'))

if __name__ == "__main__":
    mrq.args = mrq.setup_args()

    if mrq.args.listen_path is not None and mrq.args.listen_path != "/":
        import uvicorn
        uvicorn.run("main:app", host=mrq.args.listen_host, port=mrq.args.listen_port if not None else 8000)
    else:
        mrq.webui = mrq.setup_gradio()
        mrq.webui.launch(share=mrq.args.share, prevent_thread_lock=True, server_name=mrq.args.listen_host, server_port=mrq.args.listen_port)
        mrq.tts = mrq.setup_tortoise()

        mrq.webui.block_thread()
elif __name__ == "main":
    from fastapi import FastAPI
    import gradio as gr

    import sys
    sys.argv = [sys.argv[0]]

    app = FastAPI()
    mrq.args = mrq.setup_args()
    mrq.webui = mrq.setup_gradio()
    app = gr.mount_gradio_app(app, mrq.webui, path=mrq.args.listen_path)

    mrq.tts = mrq.setup_tortoise()
