from pathlib import Path
import os
from copy import copy
from omegaconf import DictConfig, OmegaConf
from modules import paths, scripts, script_callbacks, shared, images, scripts_postprocessing
from typing import TYPE_CHECKING, Any, NamedTuple
import pywt
import random
from PIL import Image, ImageFilter, ImageChops, ImageEnhance, ImageDraw, ImageFont
from translate import Translator
from langdetect import detect
import re
import numpy as np
import gradio as gr
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from modules.shared import cmd_opts, opts, state
from modules.processing import (
    Processed,
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)
from transformers import MarianMTModel, MarianTokenizer
import re


# Variables globales para mantener el modelo y el tokenizador actuales
current_model = None
tokenizer = None
model = None

#model_name = "Helsinki-NLP/opus-mt-es-en"
#tokenizer = MarianTokenizer.from_pretrained(model_name)
#model = MarianMTModel.from_pretrained(model_name)


CONFIG_PATH = Path(__file__).parent.resolve() / '../config.yaml'



print(
    f"[-] rMada ProImage initialized."
)

class RmadaScaler(torch.nn.Module):
    def __init__(self, scale, block, scaler):
        super().__init__()
        self.scale = scale
        self.block = block
        self.scaler = scaler
        
    def forward(self, x, *args):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode=self.scaler)
        return self.block(x, *args)


class RmadaUPS(scripts.Script):
    def __init__(self):
        super().__init__()
        try:
            self.config: DictConfig = OmegaConf.load(CONFIG_PATH)
        except Exception:
            self.config = DictConfig({})
        self.disable = False
        self.step_limit = 0
        self.infotext_fields = []

    def title(self):
        return "rMada ProImage"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def load_model(self, model_name):
        global current_model, tokenizer, model
        # Verifica si el modelo solicitado ya está cargado
        if current_model != model_name:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            current_model = model_name
            # print(f"Translation model loaded: {model_name}")
        #else:
            # print(f"Model {model_name} is already loaded.")

    def ui(self, is_img2img):
        with gr.Accordion(label='rMada ProImage', open=False):
            with gr.Row():
                RMADA_enable = gr.Checkbox(label='Enable extension', value=self.config.get('RMADA_enable', False))
            with gr.Tab("General"):
                with gr.Row():
                    with gr.Group():
                        RMADA_loras = gr.Checkbox(label='Enable RMSDXL Loras', value=self.config.get('RMADA_loras', False))
                        RMADA_CheckSharpen = gr.Checkbox(label='Enable Sharpen', value=self.config.get('RMADA_CheckSharpen', True))
                        RMADA_CheckEnhance = gr.Checkbox(label='Enable Enhance', value=self.config.get('RMADA_CheckEnhance', True))
                        RMADA_CheckFilters = gr.Checkbox(label='Enable Filters', value=self.config.get('RMADA_CheckFilters', True))
                        RMADA_CheckCopyright = gr.Checkbox(label='Enable Copyright', value=self.config.get('RMADA_CheckCopyright', True))
                    with gr.Group():
                        RMADA_removeloras = gr.Checkbox(label='Remove all Loras from prompt', value=self.config.get('RMADA_removeloras', False))
                        RMADA_removeemphasis = gr.Checkbox(label='Remove emphasis from prompt', value=self.config.get('RMADA_removeemphasis', False))
                        RMADA_fixprompt = gr.Checkbox(label='Fix prompt first word', value=self.config.get('RMADA_fixprompt', True))
                        RMADA_moveloras = gr.Checkbox(label='Move Loras to the end of the prompt', value=self.config.get('RMADA_moveloras', True))
                    with gr.Group():
                        RMADA_fixhr = gr.Checkbox(label='Auto Hires Fix', value=self.config.get('RMADA_fixhr', True))
                        RMADA_translate = gr.Checkbox(label='Auto Translate', value=self.config.get('RMADA_translate', False))
                        RMADA_SaveBefore = gr.Checkbox(label='Save Image Before PostPro', value=self.config.get('RMADA_SaveBefore', False))
            with gr.Tab("Sharpen"):
                with gr.Row():
                    RMADA_sharpenweight = gr.Slider(minimum=0, maximum=10, step=0.01, label="Sharpen", value=self.config.get('RMADA_sharpenweight', 0))
                    RMADA_edge_detection_sharpening = gr.Slider(minimum=0, maximum=10, step=0.5, label="Edge Detection Sharpening", value=self.config.get('RMADA_edge_detection_sharpening', 0))
                with gr.Row():
                    RMADA_wavelet_sharpening = gr.Slider(minimum=0, maximum=1, step=0.01, label="Wavelet Sharpening", value=self.config.get('RMADA_wavelet_sharpening', 0.4))
                    RMADA_adaptive_sharpened =  gr.Slider(minimum=0, maximum=1, step=0.01, label="Adaptive Sharpen", value=self.config.get('RMADA_adaptive_sharpened', 0))
            with gr.Tab("Enhance"):
                with gr.Row():
                    RMADA_contrast = gr.Slider(minimum=-1, maximum=1, step=0.01, label="Contrast", value=self.config.get('RMADA_contrast', -0.12))
                    RMADA_brightness = gr.Slider(minimum=-1, maximum=1, step=0.01, label="Brightness", value=self.config.get('RMADA_brightness', 0.23))
                with gr.Row():
                    RMADA_gamma = gr.Slider(minimum=0.6, maximum=2.2, step=0.01, label="Gamma", value=self.config.get('RMADA_gamma', 0.93))
                    RMADA_saturation = gr.Slider(minimum=-1, maximum=1, step=0.01, label="Saturation", value=self.config.get('RMADA_saturation', 0.08))
            with gr.Tab("Filters"):
                with gr.Row():
                    RMADA_noise = gr.Slider(minimum=0, maximum=0.1, step=0.001, label="Noise", value=self.config.get('RMADA_noise', 0.015))
                    RMADA_vignette = gr.Slider(minimum=0, maximum=1, step=0.01, label="Vignette", value=self.config.get('RMADA_vignette', 0.04))
            #        RMADA_vignette_overlay = gr.Slider(minimum=0, maximum=1, step=0.01, label="Vignette Overlay", value=self.config.get('RMADA_vignette_overlay', 0))
            with gr.Tab("RMSDXL Loras"):
                with gr.Row():
                    RMADA_lora_enhance = gr.Slider(minimum=-5, maximum=5, step=0.1, label="Enhance XL", value=self.config.get('RMADA_loras, RMADA_lora_enhance', 0))
                    RMADA_lora_creative = gr.Slider(minimum=-5, maximum=5, step=0.1, label="Creative XL", value=self.config.get('RMADA_lora_creative', 0))
                with gr.Row():
                    RMADA_lora_photo = gr.Slider(minimum=-5, maximum=5, step=0.1, label="Photo XL", value=self.config.get('RMADA_lora_photo', 0))
                    RMADA_lora_darkness = gr.Slider(minimum=-5, maximum=5, step=0.1, label="Darkness Cinema XL", value=self.config.get('RMADA_lora_darkness', 0))
                    RMADA_lora_details = gr.Slider(minimum=-5, maximum=5, step=0.1, label="Details XL", value=self.config.get('RMADA_lora_details', 0))
            #with gr.Tab("Utils"):
            #    with gr.Row():
            #        arc_show_calculator = gr.Button(
            #                            value="Calc",
            #                            visible=True,
            #                            variant="secondary",
            #                            elem_id="arc_show_calculator_button",
            #                        )
            
            with gr.Tab("Settings"):
                with gr.Row():
                    RMADA_translate_lang = gr.Dropdown(['es','zh','hi','ar','pt','bn','ru','ja','pa','de','jv','ko','fr','tr','vi','te','mr','it'], label='Lang From', 
                        value=self.config.get('RMADA_translate_lang', 'es'))
                    RMADA_translate_mode = gr.Dropdown(['Full translate', 'Detection blocks'], label='Mode', 
                        value=self.config.get('RMADA_translate_mode', 'Full translate'))
                with gr.Row():
                    RMADA_copyright = gr.Textbox(
                            label="Text/Copyright",
                            value=self.config.get('RMADA_copyright', "MODEL: {CHECKPOINT} | SAMPLER: {SAMPLER} | STEPS: {STEPS} | CFG Scale: {CFG} | WIDTH: {WIDTH} | HEIGHT: {HEIGHT}     -       @RMADA'24")
                        )


        ui = [RMADA_enable,RMADA_SaveBefore,RMADA_CheckSharpen,RMADA_CheckEnhance,RMADA_CheckFilters,RMADA_CheckCopyright, RMADA_sharpenweight, RMADA_edge_detection_sharpening, RMADA_wavelet_sharpening, RMADA_adaptive_sharpened, RMADA_contrast, RMADA_brightness, RMADA_saturation, RMADA_gamma, RMADA_noise, RMADA_vignette, RMADA_translate, RMADA_fixhr, RMADA_translate_lang, RMADA_translate_mode, RMADA_removeloras, RMADA_removeemphasis, RMADA_fixprompt, RMADA_moveloras, RMADA_copyright, RMADA_loras, RMADA_lora_enhance, RMADA_lora_creative, RMADA_lora_photo, RMADA_lora_darkness, RMADA_lora_details]
        for elem in ui:
            setattr(elem, "do_not_save_to_config", True)

        parameters = {
            'RMADA_enable': RMADA_enable,
            'RMADA_sharpenweight': RMADA_sharpenweight,
            'RMADA_edge_detection_sharpening': RMADA_edge_detection_sharpening,
            'RMADA_wavelet_sharpening': RMADA_wavelet_sharpening,
            'RMADA_adaptive_sharpened': RMADA_adaptive_sharpened,
            'RMADA_contrast': RMADA_contrast,
            'RMADA_brightness': RMADA_brightness,
            'RMADA_saturation': RMADA_saturation,
            'RMADA_gamma': RMADA_gamma,
            'RMADA_noise': RMADA_noise,
            'RMADA_vignette': RMADA_vignette,
            'RMADA_translate': RMADA_translate,
            'RMADA_translate_lang': RMADA_translate_lang,
            'RMADA_translate_mode': RMADA_translate_mode,
            'RMADA_fixhr': RMADA_fixhr,
            'RMADA_removeloras': RMADA_removeloras,
            'RMADA_removeemphasis':RMADA_removeemphasis,
            'RMADA_fixprompt': RMADA_fixprompt,
            'RMADA_moveloras': RMADA_moveloras,
            'RMADA_copyright': RMADA_copyright,
            'RMADA_loras': RMADA_loras,
            'RMADA_lora_enhance': RMADA_lora_enhance,
            'RMADA_lora_creative': RMADA_lora_creative,
            'RMADA_lora_photo': RMADA_lora_photo,
            'RMADA_lora_darkness': RMADA_lora_darkness,
            'RMADA_lora_details': RMADA_lora_details,
            'RMADA_CheckSharpen':RMADA_CheckSharpen,
            'RMADA_CheckEnhance':RMADA_CheckEnhance,
            'RMADA_CheckFilters':RMADA_CheckFilters,
            'RMADA_CheckCopyright':RMADA_CheckCopyright,
            'RMADA_SaveBefore':RMADA_SaveBefore
        }
        for k, element in parameters.items():
           self.infotext_fields.append((element, k))

        return ui

    def detectar_idioma(self, texto):
        return detect(texto)

    def traducir_segmentos(self, texto, seeds, lang, RMADA_translate_mode):
        
        global current_model, tokenizer, model

        #traductor = Translator()
        # traductor = Translator(to_lang="en")
        translate_final = False
        last_text = ''
        nuevo_texto = texto[0]
        res = []
        for i, text in enumerate(texto):
            text = re.sub(r',\s*,', ',', text)  # Elimina comas duplicadas
            if last_text != text: 
                gen = random.Random()
                gen.seed(seeds[i])

                if 'Detection blocks' in RMADA_translate_mode:
                    segmentos = re.split(r'([,.():])', text)  # Dividir el texto en segmentos
                    texto_traducido = []

                    for segmento in segmentos:
                        if segmento not in [',', '.', '(', ')', ':'] and not re.search(r'__.*?__', segmento) and re.search(r'[a-zA-Z]{2,}', segmento):  # Ignorar los separadores
                            deteccion = self.detectar_idioma(segmento)
                            if deteccion != 'en':  # Traducir solo si es español
                                traductor = Translator(from_lang=lang, to_lang='en')
                                old_segmento = segmento
                                segmentoParser = traductor.translate(segmento)
                                #if "MYMEMORY WARNING" not in segmentoParser:
                                #    segmento = segmentoParser
                                #    print("RMADA Translate: from ", old_segmento, " to ", segmento)
                                #else:
                                
                                # Texto de origen y destino
                                src_texts = segmento
                                tgt_texts = None  # No es necesario para la traducción

                                # Preparar los datos para el modelo
                                model_inputs = tokenizer(src_texts, return_tensors="pt", truncation=True, padding=True)
                                labels = tokenizer(text_target=tgt_texts, return_tensors="pt", truncation=True, padding=True) if tgt_texts else None
                                if labels:
                                    model_inputs["labels"] = labels["input_ids"]

                                # Traducir
                                translated = model.generate(**model_inputs, max_length=512)
                                segmentoParser = tokenizer.decode(translated[0], skip_special_tokens=True)

                                segmento = segmentoParser
                                #print("RMADA Translate: from ", old_segmento, " to ", segmento)
                                    
                                translate_final = True
                        #texto_traducido.append(segmento)
                        # Añadir un espacio antes del '(' si no es el primer elemento
                        if segmento == '(' and texto_traducido:
                            texto_traducido.append(' ' + segmento)
                        elif segmento == ')' and texto_traducido:
                            texto_traducido.append(segmento + ' ')
                        else:
                            texto_traducido.append(segmento)

                    res.append(''.join(texto_traducido))
                    nuevo_texto = ''.join(texto_traducido)
                    last_text = text
                elif 'Full translate' in RMADA_translate_mode:
                    # Texto de origen y destino
                    src_texts = text
                    tgt_texts = None  # No es necesario para la traducción

                    # Preparar los datos para el modelo
                    model_inputs = tokenizer(src_texts, return_tensors="pt", truncation=True, padding=True)
                    labels = tokenizer(text_target=tgt_texts, return_tensors="pt", truncation=True, padding=True) if tgt_texts else None
                    if labels:
                        model_inputs["labels"] = labels["input_ids"]

                    # Traducir
                    translated = model.generate(**model_inputs, max_length=512)
                    segmentoParser = tokenizer.decode(translated[0], skip_special_tokens=True)
                    res.append(''.join(segmentoParser))
                    nuevo_texto = ''.join(segmentoParser)
                    last_text = text
                    #print("RMADA Translate: from ", last_text, " to ", segmentoParser)
                    translate_final = True
                        
            else: 
                gen = random.Random()
                gen.seed(seeds[i])
                #print(nuevo_texto)
                res.append(''.join(nuevo_texto))

        if translate_final == True:
            return res
        
        return texto

        
    def removeEmphasis(self, texto):
        res = []
        for text in texto:
            text = text.replace("(", " (").replace(")", ") ")
            # Remove content in multiple parentheses, including the optional ending with :NUMBER, completely removing the :NUMBER part
            text = re.sub(r'\({1,}([^():]*?)(:\d+\.?\d*)?\){1,}', r'\1', text)

            text = text.replace("(", "").replace(")", "")
            
            # Eliminar espacios extra que podrían haber sido introducidos
            text = re.sub(r'\s{2,}', ' ', text).strip()
            
            res.append(text)
        return res
    
    def removeLoras(self, texto):
        res = []
        for text in texto:
            # Eliminar la etiqueta <lora:XXXX:X>
            text = re.sub(r'<lora:[^>]*>', '', text)

            # Limpiar espacios y comas extra
            text = re.sub(r',\s*,', ',', text)  # Elimina comas duplicadas
            text = text.strip(', ')  # Elimina comas y espacios al inicio y al final

            res.append(text)
        return res
    
    def fixPrompt(self, texto):
        res = []
        for text in texto:
            # Primero, reemplazar los saltos de línea y múltiples espacios/espacios antes de comas con una única coma
            text = re.sub(r'\s*\n\s*|\s*,\s*|\s{2,}', ', ', text.strip())
            # Luego, eliminar cualquier coma duplicada resultante, asegurando dejar solo una coma y un espacio como separador
            text = re.sub(r',\s*,+', ', ', text)
            # Finalmente, eliminar cualquier coma al final del texto
            text = re.sub(r',\s*$', '', text)
            if not text.startswith('+'):
                text = '+ ' + text
            res.append(text)
        return res
    

    def mover_etiquetas_lora(self,texto):
        res = []
        for text in texto:
            # Encuentra todas las etiquetas <lora>
            etiquetas_lora = re.findall(r'<lora:[^>]+>', text)
            
            # Elimina las etiquetas <lora> del texto original
            texto_sin_lora = re.sub(r'<lora:[^>]+>', '', text)
            
            # Concatena el texto original sin las etiquetas <lora> y todas las etiquetas <lora> encontradas
            texto_final = texto_sin_lora + ''.join(etiquetas_lora)
            res.append(texto_final)
        
        return res

    def mover_etiquetas_lora(self,texto):
        res = []
        for text in texto:
            # Encuentra todas las etiquetas <lora>
            etiquetas_lora = re.findall(r'<lora:[^>]+>', text)
            
            # Elimina las etiquetas <lora> del texto original
            texto_sin_lora = re.sub(r'<lora:[^>]+>', '', text)
            
            # Concatena el texto original sin las etiquetas <lora> y todas las etiquetas <lora> encontradas
            texto_final = texto_sin_lora + ''.join(etiquetas_lora)
            res.append(texto_final)
        
        return res
    
    def addLora(self,texto,lora,weight):
        # p.all_prompts = self.addLora(p.all_prompts,'RMSDXL_Enhance',RMADA_lora_enhance)
        res = []
        for text in texto:
            texto_final = text + '' + "<lora:" + lora + ":" + str(weight) + ">"
            #print('texto_final: ' + texto_final)
            res.append(texto_final)
        
        return res



    def extraer_y_concatenar_comentarios(self, texto):
        res = []
        res_comentarios_concatenados = []
        for text in texto:
            # Encuentra todos los comentarios
            comentarios = re.findall(r'<comment:([^>]+)>', text)

            # Elimina los comentarios del texto original
            texto_sin_comentarios = re.sub(r'<comment:[^>]+>', '', text)

            # Une todos los comentarios en un solo string, separados por ". "
            comentarios_concatenados = '. '.join(comentarios)
            res.append(texto_sin_comentarios)
            res_comentarios_concatenados.append(comentarios_concatenados)

        return res, res_comentarios_concatenados


    def process(self, p, RMADA_enable, RMADA_SaveBefore, RMADA_CheckSharpen,RMADA_CheckEnhance,RMADA_CheckFilters,RMADA_CheckCopyright, RMADA_sharpenweight, RMADA_edge_detection_sharpening, RMADA_wavelet_sharpening, RMADA_adaptive_sharpened, RMADA_contrast, RMADA_brightness, RMADA_saturation, RMADA_gamma, RMADA_noise, RMADA_vignette, RMADA_translate, RMADA_fixhr, RMADA_translate_lang, RMADA_translate_mode, RMADA_removeloras, RMADA_removeemphasis, RMADA_fixprompt, RMADA_moveloras, RMADA_copyright, RMADA_loras, RMADA_lora_enhance, RMADA_lora_creative, RMADA_lora_photo, RMADA_lora_darkness, RMADA_lora_details):
        self.config = DictConfig({name: var for name, var in locals().items() if name not in ['self', 'p']})
        self.step_limit = 0

        # print(self.t2i_w)

        if not RMADA_enable or self.disable:
            script_callbacks.remove_current_script_callbacks()
            return
        
        model = p.sd_model.model.diffusion_model
        
        p.all_prompts, res_comentarios_concatenados = self.extraer_y_concatenar_comentarios(p.all_prompts)

        if RMADA_removeemphasis:
            p.all_prompts = self.removeEmphasis(p.all_prompts)

        if RMADA_removeloras:
            p.all_prompts = self.removeLoras(p.all_prompts)
        
        original_prompt = p.all_prompts[0]
        original_prompt_negative = p.all_negative_prompts[0]

        if RMADA_fixprompt:
            p.all_prompts = self.fixPrompt(p.all_prompts)
        
        
        if RMADA_translate:

            self.load_model("Helsinki-NLP/opus-mt-"+ RMADA_translate_lang +"-en")

            # p.all_prompts[0] = self.traducir_segmentos(original_prompt)
            translate_prompt = self.traducir_segmentos(p.all_prompts, p.all_seeds, RMADA_translate_lang, RMADA_translate_mode)
            translate_prompt_negative = self.traducir_segmentos(p.all_negative_prompts, p.all_seeds, RMADA_translate_lang, RMADA_translate_mode)
            if p.all_prompts != translate_prompt: # ya no es necesario, porque se encarga de verificar si es el mismo prompt la función de traducir
                p.all_prompts = translate_prompt
            if p.all_negative_prompts != translate_prompt_negative: # ya no es necesario, porque se encarga de verificar si es el mismo prompt la función de traducir
                p.all_negative_prompts = translate_prompt_negative
            if getattr(p, 'all_hr_prompts', None) is not None:
                p.all_hr_prompts = self.traducir_segmentos(p.all_hr_prompts, p.all_seeds, RMADA_translate_lang, RMADA_translate_mode)
            if original_prompt != p.all_prompts[0]:
                p.extra_generation_params["RMADA_Translate_Original"] = original_prompt
            if original_prompt_negative != p.all_negative_prompts[0]:
                p.extra_generation_params["RMADA_Translate_Original_Negative"] = original_prompt_negative

        if RMADA_moveloras:
            p.all_prompts = self.mover_etiquetas_lora(p.all_prompts)

        #print('RMADA_lora_enhance: ' + RMADA_lora_enhance)

        if RMADA_loras:
            if RMADA_lora_enhance != 0:
                p.all_prompts = self.addLora(p.all_prompts,'RMSDXL_Enhance',RMADA_lora_enhance)
            if RMADA_lora_creative != 0:
                p.all_prompts = self.addLora(p.all_prompts,'RMSDXL_Creative',RMADA_lora_creative)
            if RMADA_lora_photo != 0:
                p.all_prompts = self.addLora(p.all_prompts,'RMSDXL_Photo',RMADA_lora_photo)
            if RMADA_lora_darkness != 0:
                p.all_prompts = self.addLora(p.all_prompts,'RMSDXL_Darkness_Cinema_v2.0',RMADA_lora_darkness)
            if RMADA_lora_details != 0:
                p.all_prompts = self.addLora(p.all_prompts,'RMSDXL_Details',RMADA_lora_details)
        
            
        self.only_one_pass = True
        self.d1 = 4
        self.d2 = 4
        self.s1 = 0.3
        self.s2 = 0.3
        self.scaler = 'bicubic'
        self.downscale = 0.65
        self.upscale = 2
        self.smooth_scaling = True
        self.early_out = False
        
        model = p.sd_model.model.diffusion_model
        if self.s1 > self.s2: self.s2 = self.s1
        self.p1 = (self.s1, self.d1 - 1)
        self.p2 = (self.s2, self.d2 - 1)
        self.step_limit = 0
            
        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            if params.sampling_step < self.step_limit: return
            for s, d in [self.p1, self.p2]:
                out_d = d if self.early_out else -(d + 1)
                if params.sampling_step < params.total_sampling_steps * s:
                    if not isinstance(model.input_blocks[d], RmadaScaler):
                        model.input_blocks[d] = RmadaScaler(self.downscale, model.input_blocks[d], self.scaler)
                        model.output_blocks[out_d] = RmadaScaler(self.upscale, model.output_blocks[out_d], self.scaler)
                    elif self.smooth_scaling:
                        scale_ratio = params.sampling_step / (params.total_sampling_steps * s)
                        downscale = min((1 - self.downscale) * scale_ratio + self.downscale, 1.0)
                        model.input_blocks[d].scale = downscale
                        model.output_blocks[out_d].scale = self.upscale * (self.downscale / downscale)
                    return
                elif isinstance(model.input_blocks[d], RmadaScaler) and (self.p1[1] != self.p2[1] or s == self.p2[0]):
                    model.input_blocks[d] = model.input_blocks[d].block
                    model.output_blocks[out_d] = model.output_blocks[out_d].block
            self.step_limit = params.sampling_step if self.only_one_pass else 0

        if RMADA_fixhr and not self.es_tamano_estandar(p.width,p.height):
            script_callbacks.on_cfg_denoiser(denoiser_callback)
        else:
            script_callbacks.remove_current_script_callbacks()

        
        if res_comentarios_concatenados:
            p.extra_generation_params['RMADA_Comments'] = res_comentarios_concatenados

        parameters = {
            'RMADA_enable': RMADA_enable,
            'RMADA_sharpenweight': RMADA_sharpenweight,
            'RMADA_edge_detection_sharpening': RMADA_edge_detection_sharpening,
            'RMADA_wavelet_sharpening': RMADA_wavelet_sharpening,
            'RMADA_adaptive_sharpened': RMADA_adaptive_sharpened,
            'RMADA_contrast': RMADA_contrast,
            'RMADA_brightness': RMADA_brightness,
            'RMADA_saturation': RMADA_saturation,
            'RMADA_gamma': RMADA_gamma,
            'RMADA_noise': RMADA_noise,
            'RMADA_vignette': RMADA_vignette,
            'RMADA_translate': RMADA_translate,
            'RMADA_translate_lang': RMADA_translate_lang,
            'RMADA_translate_mode': RMADA_translate_mode,
            'RMADA_fixhr': RMADA_fixhr,
            'RMADA_removeloras': RMADA_removeloras,
            'RMADA_removeemphasis': RMADA_removeemphasis,
            'RMADA_fixprompt': RMADA_fixprompt,
            'RMADA_moveloras': RMADA_moveloras,
            'RMADA_copyright': RMADA_copyright,
            'RMADA_loras': RMADA_loras,
            'RMADA_lora_enhance': RMADA_lora_enhance,
            'RMADA_lora_creative': RMADA_lora_creative,
            'RMADA_lora_photo': RMADA_lora_photo,
            'RMADA_lora_darkness': RMADA_lora_darkness,
            'RMADA_lora_details': RMADA_lora_details,
            'RMADA_CheckSharpen':RMADA_CheckSharpen,
            'RMADA_CheckEnhance':RMADA_CheckEnhance,
            'RMADA_CheckFilters':RMADA_CheckFilters,
            'RMADA_CheckCopyright':RMADA_CheckCopyright,
            'RMADA_SaveBefore': RMADA_SaveBefore
        }



        for k, v in parameters.items():
            if v != 0 and 'RMADA_fixhr' not in k and 'RMADA_enable' not in k and 'RMADA_SaveBefore' not in k and 'RMADA_CheckSharpen' not in k and 'RMADA_CheckEnhance' not in k and 'RMADA_CheckFilters' not in k and 'RMADA_CheckCopyright' not in k and 'RMADA_translate_lang' not in k and 'RMADA_translate' not in k and 'RMADA_removeloras' not in k and 'RMADA_removeemphasis' not in k and 'RMADA_fixprompt' not in k and 'RMADA_moveloras' not in k and 'RMADA_copyright' not in k and 'RMADA_loras' not in k and 'RMADA_lora_enhance' not in k and 'RMADA_lora_creative' not in k and 'RMADA_lora_photo' not in k and 'RMADA_lora_darkness' not in k and 'RMADA_lora_details' not in k:
                p.extra_generation_params[k] = v

    @staticmethod
    def infotext(p) -> str:
        return create_infotext(
            p, p.all_prompts, p.all_seeds, p.all_subseeds, None, 0, 0
        )

    @staticmethod
    def get_i2i_init_image(p, pp):
        if getattr(p, "_ad_skip_img2img", False):
            return p.init_images[0]
        return pp.image

    @staticmethod
    def ensure_rgb_image(image: Any):
        if not isinstance(image, Image.Image):
            image = to_pil_image(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    
    def get_seed(self, p) -> tuple[int, int]:
        i = 0

        if not p.all_seeds:
            seed = p.seed
        elif i < len(p.all_seeds):
            seed = p.all_seeds[i]
        else:
            j = i % len(p.all_seeds)
            seed = p.all_seeds[j]

        if not p.all_subseeds:
            subseed = p.subseed
        elif i < len(p.all_subseeds):
            subseed = p.all_subseeds[i]
        else:
            j = i % len(p.all_subseeds)
            subseed = p.all_subseeds[j]

        return seed, subseed

    def save_image(self, p, image, *, suffix: str) -> None:
        i = 0
        if p.all_prompts:
            i %= len(p.all_prompts)
            save_prompt = p.all_prompts[i]
        else:
            save_prompt = p.prompt
        seed, _ = self.get_seed(p)

        images.save_image(
            image=image,
            path=p.outpath_samples,
            basename="",
            seed=seed,
            prompt=save_prompt,
            extension=opts.samples_format,
            info=self.infotext(p),
            p=p,
            suffix=suffix,
        )

    def es_tamano_estandar(self, width, height):
        # Definimos una tupla de tuplas con los tamaños estándar
        tamanos_estandar = (
            (1024, 1024),
            (1152, 896),
            (896, 1152),
            (1216, 832),
            (832, 1216),
            (1344, 768),
            (768, 1344),
            (1536, 640),
            (640, 1536),
        )
        
        # Comprobamos si el tamaño proporcionado está en la lista de tamaños estándar
        if (width, height) in tamanos_estandar:
            return True
        else:
            return False
            
    def postprocess(self, p, processed, *args):
        if self.config.RMADA_fixhr and not self.es_tamano_estandar(p.width,p.height):
            for i, b in enumerate(p.sd_model.model.diffusion_model.input_blocks):
                if isinstance(b, RmadaScaler):
                    p.sd_model.model.diffusion_model.input_blocks[i] = b.block
            for i, b in enumerate(p.sd_model.model.diffusion_model.output_blocks):
                if isinstance(b, RmadaScaler):
                    p.sd_model.model.diffusion_model.output_blocks[i] = b.block
        
    def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
        
        def apply_custom_sharpen(image, intensity=1.0):
            sharpen_matrix = [
                -1, -1, -1,
                -1, 9 + (10 - intensity), -1,
                -1, -1, -1
            ]
            sharpen_filter = ImageFilter.Kernel((3, 3), sharpen_matrix)
            return image.filter(sharpen_filter)
        
        def apply_unsharp_mask(image, radius=2, percent=150, threshold=3):
            return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

        def apply_high_pass_sharpen(image, blur_radius=2, scale=1.0):
            """
            Aplica un filtro de paso alto para el enfoque.

            :param image: Imagen PIL a procesar.
            :param blur_radius: Radio de desenfoque para la versión suavizada de la imagen.
            :param scale: Escala de enfoque aplicada a la imagen de paso alto.
            :return: Imagen PIL enfocada.
            """
            # Asegurarse de procesar la imagen en color
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Crear una versión suavizada de la imagen
            blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))

            # Crear la imagen de paso alto restando la versión suavizada de la imagen original
            high_pass = ImageChops.subtract(image, blurred_image)

            # Escalar la imagen de paso alto (opcional)
            high_pass = ImageChops.multiply(high_pass, scale)

            # Sumar la imagen de paso alto a la imagen original para enfocar
            sharpened_image = ImageChops.add(image, high_pass)

            return sharpened_image

        def apply_edge_detection_sharpen(image, intensity=1.0):
            """
            Aplica un enfoque de detección de bordes a la imagen.

            :param image: Imagen PIL a procesar.
            :param intensity: Intensidad del enfoque.
            :return: Imagen PIL enfocada.
            """
            
            edge_threshold = 100
            
            # Asegurarse de procesar la imagen en color
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convertir a escala de grises para la detección de bordes
            gray_image = image.convert('L')

            # Aplicar el operador Sobel para la detección de bordes
            edges = gray_image.filter(ImageFilter.FIND_EDGES)

            # Convertir los bordes a una imagen en blanco y negro
            edges_bw = edges.point(lambda x: 0 if x < edge_threshold else 255, '1')

            # Convertir la imagen de bordes a un array y escalarla
            edge_data = np.array(edges_bw) / 255 * intensity

            # Ampliar edge_data para que tenga 3 canales
            edge_data = np.stack((edge_data, edge_data, edge_data), axis=-1)

            # Aplicar el enfoque a la imagen original basado en los bordes detectados
            original_data = np.array(image)
            
            # sharpened_data = np.clip(original_data + edge_data, 0, 255).astype(np.uint8)
            sharpened_data = np.clip(original_data + (edge_data * 255 * intensity), 0, 255).astype(np.uint8)

            # Convertir de nuevo a imagen PIL
            sharpened_image = Image.fromarray(sharpened_data)

            return sharpened_image
                
        def ajustar_gamma(imagen, gamma):
            # Convierte la imagen a RGB si no lo es
            if imagen.mode != 'RGB':
                imagen = imagen.convert('RGB')

            # Ajusta los valores de los píxeles para el valor de gamma especificado
            inversa_gamma = 1.0 / gamma
            tabla = [int((i / 255.0) ** inversa_gamma * 255) for i in range(256)]

            # Si la imagen es en escala de grises, usa una tabla LUT de 256 entradas
            if imagen.mode == 'L':
                return imagen.point(tabla)

            # Para imágenes RGB, repite la tabla LUT para cada canal de color
            return imagen.point(tabla * 3)
        

        # Definir la función de afilado wavelet con las modificaciones sugeridas
        def apply_wavelet_sharpen(image, intensity=0.5):
            """
            Aplica un afilado avanzado utilizando la transformada wavelet, con la posibilidad de ajustar la intensidad y aplicar un umbral.

            :param image: Imagen PIL a procesar.
            :param intensity: Intensidad del enfoque.
            :param threshold: Umbral para aplicar el enfoque.
            :return: Imagen PIL enfocada.
            """
            
            # umbral, dejamos 10 por defecto.
            threshold=10

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convertir la imagen a un array de NumPy
            img_array = np.array(image, dtype=float)

            # Aplicar transformada wavelet por separado en cada canal de color
            final_sharpened_arrays = []
            for channel in range(3):
                # Descomponer la imagen utilizando transformada wavelet
                coeffs = pywt.wavedec2(img_array[:, :, channel], 'sym5', level=2)
                cA, (cH1, cV1, cD1), (cH2, cV2, cD2) = coeffs

                # Aplicar enfoque a los coeficientes de alta frecuencia usando un umbral
                cH1 = np.where(np.abs(cH1) > threshold, cH1 * (1 + intensity), cH1)
                cV1 = np.where(np.abs(cV1) > threshold, cV1 * (1 + intensity), cV1)
                cD1 = np.where(np.abs(cD1) > threshold, cD1 * (1 + intensity), cD1)
                cH2 = np.where(np.abs(cH2) > threshold, cH2 * (1 + intensity/2), cH2)
                cV2 = np.where(np.abs(cV2) > threshold, cV2 * (1 + intensity/2), cV2)
                cD2 = np.where(np.abs(cD2) > threshold, cD2 * (1 + intensity/2), cD2)

                # Reconstruir la imagen a partir de los coeficientes modificados
                coeffs = cA, (cH1, cV1, cD1), (cH2, cV2, cD2)
                sharpened_channel = pywt.waverec2(coeffs, 'sym5')
                
                # Asegurarse de que los valores estén dentro del rango permitido
                sharpened_channel = np.clip(sharpened_channel, 0, 255)
                final_sharpened_arrays.append(sharpened_channel)

            # Combinar los canales de color en una sola imagen
            sharpened_img_array = np.stack(final_sharpened_arrays, axis=-1)

            # Convertir de vuelta a una imagen PIL y devolver
            sharpened_image = Image.fromarray(sharpened_img_array.astype(np.uint8))

            return sharpened_image

            
        def apply_wavelet_sharpen_back(image, intensity=0.5):
            """
            Aplica wavelet sharpening a una imagen y la fusiona con la original,
            manteniendo el espacio de color correcto.

            :param image: Imagen PIL a procesar.
            :param intensity: Intensidad del enfoque.
            :return: Imagen PIL enfocada.
            """
            # Asegurarse de procesar la imagen en color
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convertir la imagen a un array de NumPy
            img_array = np.array(image)

            # Aplicar transformada wavelet por separado en cada canal de color
            final_sharpened_arrays = []
            for channel in range(3):  # Procesar R, G y B independientemente
                # Descomponer la imagen utilizando transformada wavelet
                coeffs = pywt.wavedec2(img_array[:, :, channel], 'haar', level=1)
                cA, (cH, cV, cD) = coeffs

                # Aplicar enfoque a los coeficientes de alta frecuencia
                cH += cH * intensity
                cV += cV * intensity
                cD += cD * intensity

                # Reconstruir la imagen a partir de los coeficientes modificados
                coeffs = cA, (cH, cV, cD)
                sharpened_channel = pywt.waverec2(coeffs, 'haar')
                
                # Asegurarse de que los valores estén dentro del rango permitido
                sharpened_channel = np.clip(sharpened_channel, 0, 255)
                final_sharpened_arrays.append(sharpened_channel)

            # Combinar los canales de color en una sola imagen
            sharpened_img_array = np.stack(final_sharpened_arrays, axis=-1)

            # Convertir de vuelta a una imagen PIL y devolver
            sharpened_image = Image.fromarray(sharpened_img_array.astype(np.uint8))

            return sharpened_image

                
        def estimate_local_sharpness(image, radius=2):
            """
            Estima la nitidez local de la imagen.

            :param image: Imagen PIL a procesar.
            :param radius: Radio para el cálculo de la nitidez local.
            :return: Array de NumPy representando la nitidez local.
            """
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
            sharpness_estimate = ImageChops.subtract(image, blurred_image)
            return np.array(sharpness_estimate)

        def apply_adaptive_sharpen(image, intensity=1.0):
            """
            Aplica un enfoque adaptativo a la imagen, usando la intensidad como factor de mezcla.

            :param image: Imagen PIL a procesar.
            :param intensity: Intensidad general del enfoque, valor entre 0 y 1.
            :return: Imagen PIL enfocada.
            """
            
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Utiliza un valor fijo para 'percent' que funcione bien en la mayoría de los casos
            sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150))

            # Convertir imágenes a arrays
            original_array = np.array(image)
            sharpened_array = np.array(sharpened_image)

            # Calcula el valor de mezcla basado en la intensidad
            blend_factor = np.clip(intensity, 0, 1)

            # Mezcla la imagen original y la imagen enfocada
            mixed_array = (1 - blend_factor) * original_array + blend_factor * sharpened_array

            # Asegurarse de que los valores estén en el rango adecuado
            mixed_array = np.clip(mixed_array, 0, 255).astype(np.uint8)

            return Image.fromarray(mixed_array)
        

        def apply_adaptive_sharpen_backup(image, intensity=1.0):
            """
            Aplica un enfoque adaptativo a la imagen.

            :param image: Imagen PIL a procesar.
            :param intensity: Intensidad general del enfoque.
            :return: Imagen PIL enfocada.
            """
            
            # Asegúrate de que la imagen sea en RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            sharpness_map = estimate_local_sharpness(image)
            # sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150 * intensity))
            # tenemos que ver cómo hacer esto.... solo admite integers, y el 1 es demasiado alto.
            sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=((intensity+1)*100) * 1))

            # Convertir imágenes a arrays para combinarlas
            original_array = np.array(image)
            sharpened_array = np.array(sharpened_image)
            
            #sharpness_map = sharpness_map / sharpness_map.max()  # Normalizar
            
            # Normalizar sharpness_map
            sharpness_map = sharpness_map / sharpness_map.max()

            # Verificar que sharpness_map solo tiene dos dimensiones
            if len(sharpness_map.shape) == 2:
                sharpness_map = sharpness_map[..., np.newaxis]  # Añadir una tercera dimensión
            #print("Sharpness map shape after adding new axis:", sharpness_map.shape)
  

            # Aplicar el enfoque de manera adaptativa
            RMADA_adaptive_sharpened_array = original_array + (sharpened_array - original_array) * sharpness_map

            # Asegurarse de que los valores estén en el rango adecuado
            RMADA_adaptive_sharpened_array = np.clip(RMADA_adaptive_sharpened_array, 0, 255).astype(np.uint8)

            return Image.fromarray(RMADA_adaptive_sharpened_array)
        
        def adjust_RMADA_contrast(image, RMADA_contrast_factor=1.0):
            """
            Ajusta el RMADA_contraste de una imagen.

            :param image: Imagen PIL a procesar.
            :param RMADA_contrast_factor: Factor por el que se ajustará el RMADA_contraste.
            :return: Imagen PIL con el RMADA_contraste ajustado.
            """
            
            RMADA_contrast_factor = RMADA_contrast_factor * 0.5 + 1
            
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(RMADA_contrast_factor)
        
        def adjust_RMADA_brightness(image, RMADA_brightness_factor=1.0):
            """
            Ajusta el brillo de una imagen.

            :param image: Imagen PIL a procesar.
            :param RMADA_brightness_factor: Factor por el que se ajustará el brillo.
            :return: Imagen PIL con el brillo ajustado.
            """
            
            RMADA_brightness_factor = RMADA_brightness_factor * 0.5 + 1
            
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(RMADA_brightness_factor)
        
        def adjust_RMADA_saturation(image, RMADA_saturation_factor=1.0):
            """
            Ajusta la saturación de una imagen.

            :param image: Imagen PIL a procesar.
            :param RMADA_saturation_factor: Factor por el que se ajustará la saturación.
            :return: Imagen PIL con la saturación ajustada.
            """
            
            RMADA_saturation_factor = RMADA_saturation_factor * 0.5 + 1
            
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(RMADA_saturation_factor)

        def add_film_RMADA_noise(image, RMADA_noise_intensity=0.5):
            """
            Añade ruido de película a una imagen.

            :param image: Imagen PIL a procesar.
            :param RMADA_noise_intensity: Intensidad del ruido añadido.
            :return: Imagen PIL con ruido de película.
            """
            # Convertir la imagen a un array de NumPy
            img_array = np.array(image)

            # Generar ruido
            RMADA_noise = np.random.normal(loc=0, scale=255 * RMADA_noise_intensity, size=img_array.shape)

            # Añadir ruido a la imagen
            noisy_img_array = img_array + RMADA_noise

            # Asegurarse de que los valores estén dentro del rango permitido
            noisy_img_array = np.clip(noisy_img_array, 0, 255)

            # Convertir de vuelta a una imagen PIL
            noisy_image = Image.fromarray(noisy_img_array.astype(np.uint8))

            return noisy_image
                
        def overlay_vignette(image, strength=0.5):
            """
            Aplica un efecto de viñeta a una imagen, con un control adicional sobre la transparencia.
            :param image: Imagen PIL a procesar.
            :param strength: Determina cuán lejos hacia el centro se extiende la viñeta.
            :param transparency: Controla la opacidad máxima de la viñeta.
            :return: Imagen PIL con efecto de viñeta.
            """
            transparency=0.5
            # Crear la máscara de viñeta
            width, height = image.size
            x, y = np.ogrid[:height, :width]
            # Calcula la distancia al centro y la normaliza
            center = np.array([height / 2, width / 2])
            distance_to_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            max_distance = np.sqrt(center[0]**2 + center[1]**2)
            #normalized_distance = distance_to_center / max_distance
            # Calcula el gradiente de viñeta basado en la intensidad y la transparencia
            #vignette_gradient = 1 - normalized_distance**strength
            #vignette_gradient = vignette_gradient * transparency + (1 - transparency)
            
            # Modificar la función del gradiente para hacer el centro menos oscuro
            normalized_distance = distance_to_center / max_distance
            vignette_gradient = 1 - (normalized_distance ** (strength / 2))
            
            # Aplica la transparencia al gradiente
            vignette_gradient = vignette_gradient * transparency + (1 - transparency)
        
            
            vignette_mask = Image.fromarray((vignette_gradient * 255).astype(np.uint8), 'L')

            # Aplica la máscara de viñeta a la imagen utilizando el modo "multiply"
            vignette_image = Image.new("RGB", image.size)
            vignette_image.paste(image)
            vignette_image.putalpha(vignette_mask)
            result_image = Image.composite(vignette_image, Image.new("RGB", image.size, "black"), vignette_mask)
            return result_image

        

        def add_RMADA_vignette(image, strength=0.5):
            """
            Añade un efecto de viñeta a una imagen.

            :param image: Imagen PIL a procesar.
            :param strength: Intensidad del efecto de viñeta.
            :return: Imagen PIL con efecto de viñeta.
            """
            # Calcular el gradiente de viñeta
            width, height = image.size
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / (width / 2)
            y = (y - height / 2) / (height / 2)
            
            #gradient = 1 - np.sqrt(x**2 + y**2) * strength
            #gradient = np.clip(gradient, 0, 1)
            
            # Modificar el gradiente para que el oscurecimiento sea más suave
            # gradient = np.sqrt(x**2 + y**2)
            # gradient = np.clip(2 - strength * gradient, 0, 1)

            # Ajustar el gradiente y aplicar el factor de intensidad
            #gradient = np.sqrt(x**2 + y**2)
            #gradient = 1 - np.clip(gradient * strength * 2, 0, 1)
            
            # Modificar el gradiente para un incremento gradual desde el centro hacia los bordes
            gradient = np.sqrt(x**2 + y**2)
            gradient = np.clip(2- gradient ** (strength * 10), 0, 1)


            # Invertir el gradiente para oscurecer los bordes
            # gradient = 1 - gradient
            
            # Aplicar el gradiente a la imagen
            vignette = Image.fromarray((gradient * 255).astype(np.uint8), 'L')
            vignette = vignette.resize(image.size)

            # Combinar la imagen original con la viñeta
            return Image.composite(image, Image.new("RGB", image.size, "black"), vignette)

            # Aplicar el gradiente a la imagen
            # RMADA_vignette = Image.fromarray((gradient * 255).astype(np.uint8), 'L')
            # RMADA_vignette = RMADA_vignette.resize(image.size)
            # RMADA_vignette_image = ImageOps.colorize(RMADA_vignette, (0, 0, 0), (255, 255, 255))

            # Combinar la imagen original con la viñeta
            # return Image.composite(image, RMADA_vignette_image, RMADA_vignette)
                

        def agregar_texto_inferior(imagen, texto, p):
            
            #print(p.cfg_scale)
            #print(p.sd_model.sd_checkpoint_info.model_name)
            #print(p.steps)
            #print(p.width)
            #print(p.height)
            #print(p.sampler_name)
            
            texto = texto.replace("{CFG}", str(p.cfg_scale))
            texto = texto.replace("{CHECKPOINT}", str(p.sd_model.sd_checkpoint_info.model_name))
            texto = texto.replace("{STEPS}", str(p.steps))
            texto = texto.replace("{WIDTH}", str(p.width))
            texto = texto.replace("{HEIGHT}", str(p.height))
            texto = texto.replace("{SAMPLER}", str(p.sampler_name))
            
            # Configuración del texto y el rectángulo
            altura_rectangulo = 30
            padding_derecho = 10
            color_rectangulo = (0, 0, 0, 128)  # Color negro con 50% de transparencia
            color_texto = (255, 255, 255, 128)  # Color blanco

            directorio_script = os.path.dirname(os.path.realpath(__file__))
            ruta_fuente = os.path.join(directorio_script, "opensans.ttf")
            
            # Cargar la fuente
            try:
                fuente = ImageFont.truetype(ruta_fuente, 22)
            except IOError:
                fuente = ImageFont.load_default()

            # Crear una imagen para el rectángulo y el texto
            ancho_imagen, alto_imagen = imagen.size
            imagen_transparente = Image.new("RGBA", imagen.size, (255, 255, 255, 0))
            dibujo = ImageDraw.Draw(imagen_transparente)

            # Dibujar el rectángulo en la parte inferior
            dibujo.rectangle([(0, alto_imagen - altura_rectangulo), (ancho_imagen, alto_imagen)], fill=color_rectangulo)

            # Calcular la posición del texto y dibujarlo
            ancho_texto, altura_texto = fuente.getsize(texto)
            posicion_texto = (ancho_imagen - ancho_texto - padding_derecho, alto_imagen - altura_rectangulo + (altura_rectangulo - altura_texto) // 2)
            dibujo.text(posicion_texto, texto, font=fuente, fill=color_texto)

            # Combinar la imagen original con la imagen transparente
            imagen_final = Image.alpha_composite(imagen.convert("RGBA"), imagen_transparente)

            return imagen_final.convert("RGB")


        if self.config.RMADA_enable:
            # print("> Applying rMada ProImage ")

            pil_image = script_pp.image           
            pil_output = pil_image

            if opts.samples_save:
                if self.config.RMADA_SaveBefore:
                    script_pp.image = self.get_i2i_init_image(p, script_pp)
                    script_pp.image = self.ensure_rgb_image(script_pp.image)
                    init_image = copy(script_pp.image)
                    self.save_image(
                        p, init_image, suffix="-original"
                    )
            
            # Calcular nuevas dimensiones
            # Por ejemplo, reducir la imagen a la mitad de su tamaño
            new_width = pil_output.width * 2
            new_height = pil_output.height * 2
            new_size = (new_width, new_height)
            # Escalar la imagen
            pil_output = pil_output.resize(new_size, Image.BILINEAR)  # Puedes cambiar el filtro (NEAREST, BILINEAR, etc.)

            
            if self.config.RMADA_CheckSharpen:
                if self.config.RMADA_sharpenweight > 0:
                    # print(f"LOG RMADA_sharpenweight: {self.config.RMADA_sharpenweight}")
                    sharpened_image = apply_custom_sharpen(pil_output, self.config.RMADA_sharpenweight)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_edge_detection_sharpening > 0:
                    # print(f"LOG RMADA_edge_detection_sharpening: {self.config.RMADA_edge_detection_sharpening}")
                    sharpened_image = apply_edge_detection_sharpen(pil_output, self.config.RMADA_edge_detection_sharpening)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_wavelet_sharpening > 0:
                    # print(f"LOG RMADA_wavelet_sharpening: {self.config.RMADA_wavelet_sharpening}")
                    sharpened_image = apply_wavelet_sharpen(pil_output, self.config.RMADA_wavelet_sharpening)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_adaptive_sharpened > 0:
                    # print(f"LOG RMADA_adaptive_sharpened: {self.config.RMADA_adaptive_sharpened}")
                    sharpened_image = apply_adaptive_sharpen(pil_output, self.config.RMADA_adaptive_sharpened)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

            if self.config.RMADA_CheckEnhance:
                if self.config.RMADA_gamma != 1:
                    # print(f"LOG RMADA_gamma: {self.config.RMADA_gamma}")
                    sharpened_image = ajustar_gamma(pil_output, self.config.RMADA_gamma)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_contrast != 0:
                    # print(f"LOG RMADA_contrast: {self.config.RMADA_contrast}")
                    sharpened_image = adjust_RMADA_contrast(pil_output, self.config.RMADA_contrast)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_brightness != 0:
                    # print(f"LOG RMADA_brightness: {self.config.RMADA_brightness}")
                    sharpened_image = adjust_RMADA_brightness(pil_output, self.config.RMADA_brightness)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)
                
                if self.config.RMADA_saturation != 0:
                    # print(f"LOG RMADA_saturation: {self.config.RMADA_saturation}")
                    sharpened_image = adjust_RMADA_saturation(pil_output, self.config.RMADA_saturation)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

            if self.config.RMADA_CheckFilters:
                if self.config.RMADA_noise != 0:
                    # print(f"LOG RMADA_noise: {self.config.RMADA_noise}")
                    sharpened_image = add_film_RMADA_noise(pil_output, self.config.RMADA_noise)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

                if self.config.RMADA_vignette != 0:
                    # print(f"LOG RMADA_vignette: {self.config.RMADA_vignette}")
                    sharpened_image = add_RMADA_vignette(pil_output, self.config.RMADA_vignette)
                    pil_output = sharpened_image
                    #processed.images[i] = np.array(sharpened_image)

            if self.config.RMADA_CheckCopyright:
                if self.config.RMADA_copyright:
                    sharpened_image = agregar_texto_inferior(pil_output, self.config.RMADA_copyright, p)
                    pil_output = sharpened_image


            # Calcular nuevas dimensiones
            # Por ejemplo, reducir la imagen a la mitad de su tamaño
            new_width = pil_output.width // 2
            new_height = pil_output.height // 2
            new_size = (new_width, new_height)
            # Escalar la imagen
            pil_output = pil_output.resize(new_size, Image.NEAREST)  # Puedes cambiar el filtro (NEAREST, BILINEAR, etc.)


            #print("Tipo de pil_output:", type(pil_output))

            # Convertir la imagen PIL a un Tensor
            #transform_to_tensor = transforms.ToTensor()
            #tensor_image = transform_to_tensor(pil_output)
            #pil_output = RMScaler(0.5, tensor_image, 'nearest')
            




            pp = scripts_postprocessing.PostprocessedImage(pil_output)
            pp.info = {}
            p.extra_generation_params.update(pp.info)
            script_pp.image = pp.image
            
            #script_pp.image = self.get_i2i_init_image(p, script_pp)
            #script_pp.image = self.ensure_rgb_image(script_pp.image)
            #init_image = copy(script_pp.image)
            #self.save_image(
            #    p, init_image, suffix="-rmsdxl"
            #)

                
        OmegaConf.save(self.config, CONFIG_PATH)
        
    def process_batch(self, p, *args, **kwargs):
        self.step_limit = 0


    def after_component(self, component, **kwargs):
        #print(kwargs.get("elem_id"))
        if kwargs.get("elem_id") == "arc_show_calculator_button":
            self.t2i_w = component
        if kwargs.get("elem_id") == "txt2img_height":
            self.t2i_h = component
